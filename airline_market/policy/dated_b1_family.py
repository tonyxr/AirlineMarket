"""A.5/B0/B1/V9: explicit all-type declarations on ONE native shared Poisson family.

Delegate frozen date/demand/physical compilers and native engine execution.
No grid choice, new physical law, empirical fit or inference/planner adoption.
"""

from dataclasses import dataclass, replace

from airline_market.policy.dated_demand import (
    DatedDemandSurge,
    compile_dated_demand,
    compile_dated_hurricane,
    dated_demand_family,
)
from airline_market.policy.dated_shocks import B1DatedShock, compile_dated_shocks


@dataclass(frozen=True)
class DatedB1Declaration:
    """One named prospective member; caller supplies every date/magnitude/mix.

    An optional physical event and optional demand declaration form exactly the
    five B1 types plus no-shock control. Only weather+surge is a B1 composite.
    """

    scenario_id: str
    physical: B1DatedShock | None = None
    demand: DatedDemandSurge | None = None

    def __post_init__(self):
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("explicit B1 scenario ID required")
        if self.physical is not None and type(self.physical) is not B1DatedShock:
            raise TypeError("explicit B1DatedShock physical declaration required")
        if self.demand is not None and type(self.demand) is not DatedDemandSurge:
            raise TypeError("explicit DatedDemandSurge declaration required")
        if (
            self.physical is not None
            and self.demand is not None
            and self.physical.kind != "weather"
        ):
            raise ValueError(
                "B1 composite is weather plus surge, not an inferred new shock type"
            )

    @property
    def kind(self):
        if self.demand is not None:
            return "demand_surge" if self.physical is None else "hurricane"
        return "control" if self.physical is None else self.physical.kind

    def record(self):
        return {
            "scenario_id": self.scenario_id,
            "kind": self.kind,
            "physical": None if self.physical is None else self.physical.record(),
            "demand": None if self.demand is None else self.demand.record(),
        }


def compile_dated_b1_member(compiled, clock, departures, declaration):
    """Pure dated projection, before any tape generation or engine construction.

    Physical-only members keep EXACT baseline compiled demand and empty demand
    periods/witnesses, but carry their physical kind/reveal and original design.
    Frozen physical compilation enforces the actual per-leg/time contracts.
    """
    if type(declaration) is not DatedB1Declaration:
        raise TypeError("DatedB1Declaration required")
    d = declaration
    departures = tuple(departures)
    if d.kind == "hurricane":
        return compile_dated_hurricane(
            compiled,
            clock,
            departures,
            scenario_id=d.scenario_id,
            capacity=d.physical,
            demand=d.demand,
        )
    native = compile_dated_demand(
        compiled, clock, departures, d.demand, control_id=d.scenario_id
    )
    if d.physical is None:
        return replace(native, scenario_id=d.scenario_id)
    (mapping,) = compile_dated_shocks(compiled, clock, departures, (d.physical,))
    return replace(
        native,
        scenario_id=d.scenario_id,
        kind=d.kind,
        physical_shocks=(d.physical,),
        reveal=mapping.reveal,
    )


@dataclass(frozen=True)
class DatedB1Family:
    declarations: tuple
    native_family: object

    @property
    def scenarios(self):
        return self.native_family.scenarios

    @property
    def bank(self):
        return self.native_family.bank

    def engine(self, name, controllers=None, **kwargs):
        """Delegate one native engine; physical members require dated re-entry.

        The wrapper never steps, reprices, resolves or redraws a physical tape.
        Different controller/policy choices are not authenticated B2 channels.
        """
        d = dict(self.declarations)[name]
        if d.physical is not None and kwargs.get("dated_rebooking") is None:
            raise ValueError(
                "physical B1 family member requires explicit dated stranded re-entry"
            )
        return self.native_family.engine(name, controllers, **kwargs)

    def record(self):
        return {
            "plan_id": "A.5/B0/B1/V9",
            "declared_members": {k: d.record() for k, d in self.declarations},
            "native_family": self.native_family.record(),
            "coupling_scope": "ONE declared shared potential schema/family; no cross-regenerated-bank or outcome/variance equivalence",
            "physical_dated_reentry_required": True,
            "full_B1_grid_or_split_adopted": False,
            "economic_B2_channels_authenticated": False,
            "empirical_or_inference_planner_adoption": False,
            "scientific_budget": None,
        }


def dated_b1_family(compiled, clock, departures, declarations, *, seed, namespace):
    """Compile all explicit members BEFORE ONE delegated native envelope draw.

    No default list, missing third magnitude, duration/mix, fold, seed range or
    shock grid is inferred. Bank generation requires native Poisson demand and
    preserves the frozen shared family's mark/schema restrictions.
    """
    declarations = tuple(declarations)
    if not declarations or any(type(d) is not DatedB1Declaration for d in declarations):
        raise TypeError("nonempty explicit DatedB1Declaration family required")
    if len({d.scenario_id for d in declarations}) != len(declarations):
        raise ValueError("distinct explicit B1 scenario IDs required")
    items = tuple(sorted((d.scenario_id, d) for d in declarations))
    # Consume a possibly one-shot public departure iterable exactly once.
    departures = tuple(departures)
    scenarios = {
        k: compile_dated_b1_member(compiled, clock, departures, d) for k, d in items
    }
    family = dated_demand_family(scenarios, seed=seed, namespace=namespace)
    return DatedB1Family(items, family)
