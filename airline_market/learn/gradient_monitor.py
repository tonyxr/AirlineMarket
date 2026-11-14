"""Non-mutating native actor-gradient probes and declared TR5 scalar monitoring."""

import csv
import hashlib
import math
import struct
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from airline_market.core.rng import StreamRegistry
from airline_market.learn.structured_runtime import StructuredPPOAgent


def actor_loss(agent, prepared, indices):
    """Native flat transition weighting or structured episode weighting, unchanged."""
    structured = isinstance(agent, StructuredPPOAgent)
    terms, logs, old, advantages, entropies = [], [], [], [], []
    for index in indices:
        row = prepared[int(index)]
        dist, _, _ = agent.actor.distribution(row.context[:, :-1])
        logp = dist.log_prob(row.coordinates[None]).sum(-1)[0]
        entropy = dist.entropy().sum(-1)
        if structured:
            loss = agent.clipped_policy_loss(
                logp, row.old_log_probabilities, row.advantages, agent.config.clip_ratio
            )[0]
            term = loss - agent.config.entropy_coefficient * entropy.mean()
            if agent.actor.aux_head is not None:
                _, features, _ = agent.actor.features(row.context[:, :-1])
                labels = agent._episodes[int(index)].episode.metadata[
                    "joint_event_labels"
                ]
                target = torch.as_tensor(
                    labels, dtype=torch.float32, device=agent.device
                )[None]
                term = term + agent.actor.aux_head.loss(
                    features, target, agent.aux_weight
                )
            terms.append(term)
        else:
            logs.append(logp)
            old.append(row.old_log_probabilities)
            advantages.append(row.advantages)
            entropies.append(entropy)
    if structured:
        return torch.stack(terms).mean()
    loss = agent.clipped_policy_loss(
        torch.cat(logs), torch.cat(old), torch.cat(advantages), agent.config.clip_ratio
    )[0]
    return loss - agent.config.entropy_coefficient * torch.cat(entropies).mean()


def loss_gradient(agent, prepared, indices):
    parameters = tuple(agent.actor.parameters())
    loss = actor_loss(agent, prepared, indices)
    gradient = torch.autograd.grad(loss, parameters, allow_unused=True)
    vector = torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
            for p, g in zip(parameters, gradient)
        ]
    )
    if not torch.isfinite(vector).all() or not torch.isfinite(loss):
        raise FloatingPointError("nonfinite diagnostic actor gradient")
    return vector.cpu().numpy().copy(), float(loss.detach())


def gradient_repeats(agent, *, update, repeats=32, minibatch=None, progress=None):
    """Freeze full-batch GAE; independent uniform subsets, distinct episodes within."""
    if repeats != 32 or not agent._episodes:
        raise ValueError(
            "registered gradient diagnostic requires32 repeats and a rollout pool"
        )
    probe = deepcopy(agent)
    prepared = probe._prepare_episodes()
    size = probe.config.minibatch_episodes if minibatch is None else int(minibatch)
    if not 1 <= size <= len(prepared):
        raise ValueError("gradient minibatch must fit the rollout pool")
    rng = StreamRegistry(773, "analysis/D4/gradient-minibatches-v2").generator(update)
    indices = np.stack(
        [rng.choice(len(prepared), size=size, replace=False) for _ in range(repeats)]
    )
    rows, losses = [], []
    for number, draw in enumerate(indices):
        gradient, loss = loss_gradient(probe, prepared, draw)
        rows.append(gradient)
        losses.append(loss)
        if progress:
            progress(
                {
                    "draw": number,
                    "indices": draw.tolist(),
                    "loss": loss,
                    "gradient_squared_norm": float(
                        gradient.astype(np.float64) @ gradient.astype(np.float64)
                    ),
                    "gradient_sha256": hashlib.sha256(gradient.tobytes()).hexdigest(),
                }
            )
    gradients = np.stack(rows)
    variances = gradients.astype(np.float64).var(axis=0, ddof=1)
    mean = gradients.astype(np.float64).mean(axis=0)
    return {
        "gradients": gradients,
        "indices": indices,
        "losses": np.asarray(losses),
        "trace_covariance": float(variances.sum()),
        "mean_parameter_variance": float(variances.mean()),
        "mean_gradient_squared_norm": float(mean @ mean),
        "parameters": gradients.shape[1],
        "pool_episodes": len(prepared),
        "repeats": 32,
        "minibatch_episodes": size,
        "scope": "Conditional full-actor-loss pre-clipping covariance trace on a fixed fresh on-policy batch. Entropy and any declared R3aux included; R2 supplied by native full-batch preparation. Not gradient norm or population/F9variance.",
    }


@torch.no_grad()
def rollout_metrics(agent, prepared):
    """Post-update policy diagnostics under frozen rollout normalization contexts."""
    entropy, kl, clipped, targets, predicted = [], [], [], [], []
    for row in prepared:
        distribution, _, _ = agent.actor.distribution(row.context[:, :-1])
        logp = distribution.log_prob(row.coordinates[None]).sum(-1)[0]
        difference = logp - row.old_log_probabilities
        entropy.append(float(distribution.entropy().sum(-1).mean()))
        kl.append(float(-difference.mean()))
        clipped.append(
            float(
                (torch.abs(torch.exp(difference) - 1) > agent.config.clip_ratio)
                .float()
                .mean()
            )
        )
        values, _ = agent.value(row.context[:, :-1])
        predicted.extend(values[0, :, 0].cpu().tolist())
        targets.extend(row.returns.cpu().tolist())
    target, prediction = np.asarray(targets), np.asarray(predicted)
    variance = float(target.var())
    return {
        "entropy": float(np.mean(entropy)),
        "approx_kl": float(np.mean(kl)),
        "clip_fraction": float(np.mean(clipped)),
        "explained_variance": None
        if variance == 0
        else float(1 - (target - prediction).var() / variance),
        "explained_variance_defined": variance != 0,
    }


class MonitorState:
    """Twenty consecutive signed native KL estimates >.1; one half-LR restart."""

    def __init__(self, high_kl=0, restarts=0):
        self.high_kl, self.restarts = int(high_kl), int(restarts)

    def inspect(self, metrics):
        if any(
            isinstance(v, (float, np.floating)) and not math.isfinite(v)
            for v in metrics.values()
        ):
            return "nonfinite"
        self.high_kl = self.high_kl + 1 if metrics["approx_kl"] > 0.1 else 0
        return "persistent_KL" if self.high_kl >= 20 else None

    def restart(self, agent):
        if self.restarts >= 1:
            raise FloatingPointError("TR5 restart allowance exhausted")
        for optimizer in [agent.actor_optimizer, agent.value_optimizer]:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        self.restarts += 1
        self.high_kl = 0

    def state(self):
        return {"high_kl": self.high_kl, "restarts": self.restarts}


def crc32c(data):
    """Castagnoli CRC for the public TFRecord wire format (no new dependency)."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result) + bytes([value])


def _bytes_field(tag, value):
    return _varint(tag * 8 + 2) + _varint(len(value)) + value


def _record(data):
    size = struct.pack("<Q", len(data))

    def masked(x):
        c = crc32c(x)
        return struct.pack("<I", (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF)

    return size + masked(size) + data + masked(data)


class MetricLog:
    """CSV plus scalar Event/Summary TFRecords, not a TensorBoard UI claim."""

    fields = (
        "update",
        "cohorts",
        "return_mean",
        "validation_objective",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "value_loss",
        "explained_variance",
        "gradient_variance",
        "learning_rate",
        "cohorts_per_second",
        "attempt",
    )

    def __init__(self, directory):
        self.directory = Path(directory)
        self.csv = self.directory / "metrics.csv"
        self.events = self.directory / "events.out.tfevents.airline-scalars"
        if self.csv.exists() or self.events.exists():
            raise ValueError("refuse to overwrite an existing metric log")
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.csv.open("w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(self.fields)
        self.events.write_bytes(_record(_bytes_field(3, b"brain.Event:2")))

    def write(self, row):
        if any(
            isinstance(v, (float, np.floating)) and not math.isfinite(v)
            for v in row.values()
        ):
            raise FloatingPointError("nonfinite scalar rejected before logging")
        with self.csv.open("a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(
                [row.get(k) for k in self.fields]
            )
        values = []
        for name in self.fields:
            value = row.get(name)
            if value is None or name == "update":
                continue
            scalar = (
                _bytes_field(1, name.encode())
                + b"\x15"
                + struct.pack("<f", float(value))
            )
            values.append(_bytes_field(1, scalar))
        event = (
            b"\x09"
            + struct.pack("<d", time.time())
            + b"\x10"
            + _varint(int(row["update"]))
            + _bytes_field(5, b"".join(values))
        )
        with self.events.open("ab") as f:
            f.write(_record(event))
