"""Executable recurrent opportunity-cost policies and their PPO updates (R1–R3).

The Gaussian action is a vector of leg preactivations and bounded residual
coefficients. The environment executes its incidence-based, cent-rounded prices.
Likelihood ratios are calculated in the sampled coordinates, before that map.
"""

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from ..core.hashing import schema_hash
from ..core.money import round_fare
from ..core.types import PriceAction
from ..pomdp.observation import Vectorizer
from ..vendor.ride_response.recurrent.common import ActionTransform
from ..vendor.ride_response.recurrent.ppo import RecurrentPPOAgent
from .aux_belief import JointEventHead
from .structured import StructuredActorCritic


@dataclass(frozen=True)
class StructuredLayout:
    """Record the public feature and action partition used by both recurrent paths."""

    global_dim: int
    n_legs: int
    n_coefficients: int
    leg_dim: int = 7

    @property
    def observation_dim(self):
        """Return global public features followed by the shared leg rows."""
        return self.global_dim + self.n_legs * self.leg_dim


class StructuredAdapter:
    """Encode permitted observations and map sampled controls to network prices."""

    leg_names = ("inventory_fraction", "capacity_per_100", "clock", "quote_fare",
                 "quote_available", "quote_missing", "quote_seats_per_10")

    def __init__(self, env):
        """Precompute public topology links; do not inspect simulator hidden state."""
        self.env = env
        self.c = env.compiled
        self.vectorizer = Vectorizer(self.c, env.arm)
        self.keep = [k for k, name in enumerate(self.vectorizer.names())
                     if not name.startswith("inventory/")]
        self.legs = self.c.legs_of[env.i]
        self.products = self.c.products_of[env.i]
        self.layout = StructuredLayout(len(self.keep), len(self.legs),
                                       len(env.pricing.theta_bounds))
        self.quote_links = []
        for leg_index in self.legs:
            leg = self.c.spec.legs[leg_index]
            self.quote_links.append({j for j, p in enumerate(self.c.spec.products)
                                     if p.carrier != env.focal and any(
                                         (self.c.spec.legs[self.c.leg_index[x]].origin,
                                          self.c.spec.legs[self.c.leg_index[x]].dest)
                                         == (leg.origin, leg.dest) for x in p.legs)})
        n = self.layout.n_legs
        k = self.layout.n_coefficients
        bounds = np.asarray(env.pricing.theta_bounds)
        self.action_transform = ActionTransform(
            tuple(f"leg_preactivation/{self.c.spec.legs[x].id}" for x in self.legs)
            + tuple(f"theta/{j}" for j in range(k)),
            ("identity",) * n + ("bounded",) * k,
            (0.0,) * n + tuple(bounds[:, 0]),
            (1.0,) * n + tuple(bounds[:, 1]),
        )
        self.policy_class = "network_opportunity_cost_recurrent"

    def names(self):
        """Return a stable observation schema, including leg normalizations."""
        return tuple(self.vectorizer.names()[k] for k in self.keep) + tuple(
            f"leg/{self.c.spec.legs[l].id}/{name}"
            for l in self.legs for name in self.leg_names)

    def schema_hash(self):
        """Fingerprint actor information, normalizations and pricing class."""
        return schema_hash((self.names(), self.policy_class))

    def encode(self, obs):
        """Use own resources and only quotes actually present in the supplied view."""
        global_values = self.vectorizer(obs)[self.keep]
        rows = []
        for k, linked in enumerate(self.quote_links):
            quotes = [q for q in obs.quotes if q.product in linked]
            fare = [q.fare / self.c.spec.products[q.product].reference_fare
                    for q in quotes if q.fare is not None and not q.missing]
            seats = [q.seats / 10 for q in quotes
                     if q.seats is not None and not q.missing]
            rows.extend((obs.own_inventory[k] / max(1, obs.own_capacity[k]),
                         obs.own_capacity[k] / 100,
                         obs.t / self.c.spec.horizon.n_periods,
                         float(np.mean(fare)) if fare else 0.0,
                         float(np.mean([q.available is True for q in quotes]))
                         if quotes else 0.0,
                         float(np.mean([q.missing for q in quotes])) if quotes else 1.0,
                         float(np.mean(seats)) if seats else 0.0))
        return np.asarray([*global_values, *rows], dtype=np.float32)

    def price_action(self, physical, obs):
        """Add nonnegative leg opportunity costs to the existing residual price basis."""
        n = self.layout.n_legs
        pi = np.logaddexp(0.0, np.asarray(physical[:n]))
        residual = np.empty(len(self.products), dtype=np.float64)
        # The residual price rule rounds/clips, so rebuild the unrounded basis:
        # clipping before adding opportunity costs would change the R1 price map.
        spec = self.env.pricing
        markets = tuple(dict.fromkeys(self.c.spec.products[j].market for j in self.products))
        width = len(spec.controllable_features)
        for local, j in enumerate(self.products):
            p = self.c.spec.products[j]
            block = (0 if spec.sharing == "carrier" else local
                     if spec.sharing == "product" else markets.index(p.market))
            base = spec.base.get(p.id, p.reference_fare)
            base += spec.time_ramp * obs.t / (self.c.spec.horizon.n_periods - 1)
            base += sum(w * self.env.price_rule.evaluator(f, j, obs)
                        for f, w in spec.fixed_features)
            base += sum(physical[n + block * width + k]
                        * self.env.price_rule.evaluator(f, j, obs)
                        for k, f in enumerate(spec.controllable_features))
            price = base + float(pi @ self.c.A[self.env.i][:, local])
            residual[local] = round_fare(min(p.fare_cap, max(p.fare_floor, price)))
        return PriceAction(dict(zip(self.products, residual)))

    def reset(self, seed):
        """Reset only the physical cohort; recurrence is reset by the collector."""
        self.obs, _ = self.env.reset(seed)
        return self.encode(self.obs)

    def step(self, physical):
        """Execute structured prices without adding any hidden actor input."""
        self.obs, reward, done, info = self.env.step(self.price_action(physical, self.obs))
        return self.encode(self.obs), reward, done, info


class StructuredNetwork(nn.Module):
    """Shared leg representation plus a leg-permutation-invariant global recurrence."""

    def __init__(self, layout, hidden, n_events=0):
        """Keep actor and critic instances separate, as in the existing PPO trainer."""
        super().__init__()
        self.layout = layout
        global_width = layout.global_dim + layout.leg_dim + 3 + layout.n_coefficients + 2
        self.gru = nn.GRU(global_width, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.structure = StructuredActorCritic(layout.leg_dim, hidden,
                                               layout.n_coefficients, hidden)
        self.log_std = nn.Parameter(torch.full((1 + layout.n_coefficients,), -1.0))
        self.aux_head = JointEventHead(hidden, n_events) if n_events else None
        self.register_buffer("input_mask", torch.ones(
            layout.observation_dim + layout.n_legs + layout.n_coefficients + 2))

    def features(self, context, hidden=None):
        """Pool unordered legs and their previous actions before recurrent processing."""
        context = context * self.input_mask
        d, n, f = self.layout.global_dim, self.layout.n_legs, self.layout.leg_dim
        legs = context[..., d:self.layout.observation_dim].reshape(*context.shape[:-1], n, f)
        previous = context[..., self.layout.observation_dim:]
        leg_actions = previous[..., :n]
        pooled = torch.stack((leg_actions.mean(-1), leg_actions.amin(-1),
                              leg_actions.amax(-1)), -1)
        inputs = torch.cat((context[..., :d], legs.mean(-2), pooled,
                            previous[..., n:]), -1)
        global_features, state = self.gru(inputs, hidden)
        return legs, self.norm(global_features), state

    def distribution(self, context, hidden=None):
        """Sample Gaussian leg preactivations and residual coefficient coordinates."""
        legs, global_features, state = self.features(context, hidden)
        embeddings = self.structure.leg_encoder(legs)
        mean = torch.cat((self.structure.dual_head(embeddings).squeeze(-1),
                          self.structure.residual_head(global_features)), -1)
        log_std = torch.cat((self.log_std[:1].expand(self.layout.n_legs),
                             self.log_std[1:])).clamp(-5.0, 1.0).expand_as(mean)
        return Normal(mean, log_std.exp()), state, log_std


class StructuredValue(StructuredNetwork):
    """Add a global recurrent value and a sum of shared leg values."""

    def forward(self, context, hidden=None):
        """Return the additive critic value without privileged simulator inputs."""
        legs, global_features, state = self.features(context, hidden)
        value = self.structure.global_value(global_features)
        value += self.structure.leg_value(self.structure.leg_encoder(legs)).sum(-2)
        return value, state


class StructuredPPOAgent(RecurrentPPOAgent):
    """Train the R1 network, optionally with R2 credit and R3 prediction loss."""

    def __init__(self, adapter, config, *, auxiliary=False, aux_weight=0.3):
        """Reuse rollout contracts, GAE, normalization and exact behavior-logp checks."""
        super().__init__(adapter.layout.observation_dim, adapter.action_transform, config)
        self.actor = StructuredNetwork(adapter.layout, config.hidden_dim,
                                        n_events=int(auxiliary)).to(self.device)
        self.value = StructuredValue(adapter.layout, config.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.value_learning_rate)
        self.aux_weight = aux_weight

    def update(self):
        """Optimize clipped PPO, additive values, and declared joint-event BCE jointly."""
        prepared = self._prepare_episodes()
        if not prepared:
            raise RuntimeError("no structured on-policy episodes")
        labels = [x.episode.metadata.get("joint_event_labels") for x in self._episodes]
        metrics = []
        order = np.arange(len(prepared))
        stopped_early = False
        for epoch in range(self.config.update_epochs):
            self.rng.shuffle(order)
            epoch_kls = []
            for start in range(0, len(order), self.config.minibatch_episodes):
                policy_terms, value_terms, auxiliary_terms = [], [], []
                for index in order[start:start + self.config.minibatch_episodes]:
                    row = prepared[index]
                    distribution, _, _ = self.actor.distribution(row.context[:, :-1])
                    logp = distribution.log_prob(row.coordinates[None]).sum(-1)[0]
                    epoch_kls.append(float((row.old_log_probabilities - logp).mean().detach()))
                    loss, _, _ = self.clipped_policy_loss(
                        logp, row.old_log_probabilities, row.advantages, self.config.clip_ratio)
                    policy_terms.append(loss - self.config.entropy_coefficient
                                        * distribution.entropy().sum(-1).mean())
                    prediction, _ = self.value(row.context[:, :-1])
                    value = prediction[0, :, 0]
                    clipped = row.old_values + (value - row.old_values).clamp(
                        -self.config.value_clip, self.config.value_clip)
                    errors = (value - row.returns).square()
                    if self.config.value_clip > 0:
                        errors = torch.maximum(errors, (clipped - row.returns).square())
                    value_terms.append(0.5 * self.config.value_coefficient * errors.mean())
                    if self.actor.aux_head is not None:
                        _, features, _ = self.actor.features(row.context[:, :-1])
                        target = torch.as_tensor(labels[index], dtype=torch.float32,
                                                 device=self.device)[None]
                        auxiliary_terms.append(self.actor.aux_head.loss(features, target,
                                                                        self.aux_weight))
                actor_loss = torch.stack(policy_terms).mean()
                aux = torch.stack(auxiliary_terms).mean() if auxiliary_terms else actor_loss * 0
                self.actor_optimizer.zero_grad(set_to_none=True)
                (actor_loss + aux).backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
                self.actor_optimizer.step()
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss = torch.stack(value_terms).mean()
                value_loss.backward()
                value_grad = nn.utils.clip_grad_norm_(self.value.parameters(), self.config.gradient_clip)
                self.value_optimizer.step()
                values = [float(x.detach()) for x in (actor_loss, value_loss, aux, actor_grad, value_grad)]
                if not np.isfinite(values).all():
                    raise FloatingPointError("nonfinite structured PPO update")
                metrics.append(values)
            if (self.config.target_kl is not None
                    and np.mean(epoch_kls) > self.config.target_kl):
                stopped_early = True
                break
        count = len(self._episodes)
        transitions = self.pending_transitions
        observations = np.concatenate([x.episode.observations[:-1] for x in self._episodes])
        self.observation_normalizer.update(observations)
        self.reset_rollout()
        self.updates += 1
        return dict(zip(("actor_loss", "value_loss", "aux_loss", "actor_grad", "value_grad"),
                        np.mean(metrics, axis=0))) | {"episodes": count, "transitions": transitions,
                                                       "updates": self.updates,
                                                       "epochs_completed": epoch + 1,
                                                       "stopped_early": stopped_early}
