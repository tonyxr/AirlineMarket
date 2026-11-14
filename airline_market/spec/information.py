"""Derive rival feeder information from physical resources (Eq. 28)."""

from dataclasses import replace
from .model import InformationArmSpec, ShoppingRequestSpec, QuoteSpec


def derive_information_arms(
    compiled, observer: str, controlled_markets=None, *, lag: int | None = None
):
    """Find direct rivals and shared-leg feeders from the resource graph (Eqs. 1–19)."""
    s = compiled.spec
    markets = set(
        controlled_markets
        or [p.market for p in s.products if p.carrier == observer and p.controlled]
    )
    direct = [p for p in s.products if p.carrier != observer and p.market in markets]
    linked = [
        p
        for p in s.products
        if p.carrier != observer
        and p.market not in markets
        and any(p.carrier == q.carrier and set(p.legs) & set(q.legs) for q in direct)
    ]
    unrelated = [
        p.id
        for p in s.products
        if p.carrier != observer
        and p not in direct
        and p not in linked
        and not any(
            p.carrier == q.carrier and set(p.legs) & set(q.legs) for q in direct
        )
    ]
    unrelated += [b.id for b in s.background if b.carrier != observer]
    count = len(linked)
    chosen = unrelated[:count]
    request_by_target = {
        tuple(q.target_products): q.id
        for q in s.quotes.requests
        if q.observer == observer
    }

    def qid(target):
        """
        Resolve the existing or deterministic new shopping-request identifier (Eqs.
        1–19).
        """
        return request_by_target.get((target,), "q_" + target)

    ids = [qid(p.id) for p in direct]
    arms = {
        name: InformationArmSpec(
            name=name,
            observer=observer,
            requests=tuple(req),
            privileged=priv,
            derived=True,
        )
        for name, req, priv in [
            ("direct", ids, False),
            ("linked", ids + [qid(p.id) for p in linked], False),
            ("unrelated", ids + [qid(p) for p in chosen], False),
            ("oracle", ids, True),
        ]
    }
    return arms


def add_derived_arms(
    compiled, observer: str, controlled_markets=None, *, lag: int | None = None
):
    """
    Attach the derived requests and declared lag to an immutable specification (Eqs.
    1–19).
    """
    s = compiled.spec
    arms = derive_information_arms(compiled, observer, controlled_markets, lag=lag)
    requests = list(s.quotes.requests)
    existing = {q.id for q in requests}
    if lag is not None:
        selected = {qid for arm in arms.values() for qid in arm.requests}
        requests = [
            replace(q, lag_periods=lag)
            if q.observer == observer and q.id in selected
            else q
            for q in requests
        ]
    targets = {p.id: p.carrier for p in s.products}
    targets.update({b.id: b.carrier for b in s.background})
    for arm in arms.values():
        for qid in arm.requests:
            if qid not in existing:
                product = qid.removeprefix("q_")
                owner = targets[product]
                qlag = (
                    lag
                    if lag is not None
                    else (
                        0
                        if s.timing.mode == "leader_follower"
                        and owner in s.timing.leaders
                        else 1
                    )
                )
                requests.append(
                    ShoppingRequestSpec(
                        id=qid,
                        observer=observer,
                        target_products=(product,),
                        lag_periods=qlag,
                    )
                )
                existing.add(qid)
    info = tuple(a for a in s.information if a.observer != observer) + tuple(
        arms.values()
    )
    shortfall = len(arms["linked"].requests) - len(arms["unrelated"].requests)
    metadata = {**s.metadata, "placebo_shortfall": max(0, shortfall)}
    return replace(
        s,
        quotes=QuoteSpec(requests=tuple(requests)),
        information=info,
        metadata=metadata,
    )
