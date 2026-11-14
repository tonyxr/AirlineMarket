"""Operational capacity events, followed by whole-booking resolution (Eq. 15)."""

from .ledger import resolve


def refunds_possible(compiled):
    """Check reachability of a capacity-reducing event over the horizon."""
    from .exogenous import transition_row

    spec = compiled.spec
    reachable = {k for k, p in enumerate(spec.exogenous.initial) if p > 0}
    for t in range(spec.horizon.n_periods):
        following = set()
        for before in reachable:
            for after, p in enumerate(transition_row(compiled, before, t)):
                if not p:
                    continue
                following.add(after)
                names = (
                    spec.exogenous.states[before].name,
                    spec.exogenous.states[after].name,
                )
                for event in spec.capacity_process.events:
                    if not any(delta < 0 for _, delta in event.effects):
                        continue
                    trigger = event.trigger
                    if (
                        "exo_transition" in trigger
                        and tuple(trigger["exo_transition"]) == names
                    ):
                        return True
                    if (
                        "hazard" in trigger
                        and trigger["hazard"].get(
                            names[1], trigger["hazard"].get("probability", 0.0)
                        )
                        > 0
                    ):
                        return True
        reachable = following
    return False


def apply_events(world, compiled, exo_before: int, exo_after: int, t: int, tape):
    """
    Apply triggered capacity changes and resolve whole itineraries consistently (Eqs.
    3–16).
    """
    names = compiled.spec.exogenous.states
    records = []
    for index, event in enumerate(compiled.spec.capacity_process.events):
        if event.once and index in world.events_fired:
            continue
        trigger = event.trigger
        fires = False
        if "exo_transition" in trigger:
            fires = tuple(trigger["exo_transition"]) == (
                names[exo_before].name,
                names[exo_after].name,
            )
        elif "hazard" in trigger:
            hazard = trigger["hazard"]
            p = hazard.get(names[exo_after].name, hazard.get("probability", 0.0))
            fires = tape.capacity_uniforms[t, index] < p
        if not fires:
            continue
        affected = set()
        for leg, delta in event.effects:
            l = compiled.leg_index[leg]
            i = int(compiled.leg_carrier[l])
            local = compiled.legs_of[i].index(l)
            world.carriers[i].capacity[local] = max(
                0, int(world.carriers[i].capacity[local]) + int(delta)
            )
            affected.add(i)
        for i in sorted(affected):
            resolve(world, i, compiled)
        world.events_fired.add(index)
        records.append(
            {"event": index, "period": t, "carriers": tuple(sorted(affected))}
        )
    return records
