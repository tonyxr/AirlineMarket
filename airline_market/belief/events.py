"""Declared joint availability events, with an explicit no-further-sales projection."""

from dataclasses import dataclass


def rival_inventory(hidden, compiled, focal, rival):
    """
    Recover one rival's residual leg inventory from its capacity/count ledger (Eqs.
    24–25, 30).
    """
    i = compiled.carrier_index[rival] if isinstance(rival, str) else rival
    r = [k for k in range(len(compiled.spec.carriers)) if k != focal].index(i)
    cs = hidden.rivals[r]
    return i, tuple(
        cs.capacity[l]
        - sum(int(compiled.A[i][l, j]) * n for j, n in enumerate(cs.counts))
        for l in range(len(cs.capacity))
    )


@dataclass(frozen=True)
class ItinerarySellable:
    """
    Joint all-leg availability event under an explicit frozen projection (Eqs. 24–25,
    30).
    """

    rival: str | int
    product: str | int
    at_period: int | None = None
    projection: str = "no_further_sales"

    def evaluate(self, h, c, focal):
        """Test this declared event against one model hypothesis (Eqs. 24–25, 30)."""
        if self.projection != "no_further_sales":
            raise ValueError("declare an implemented projection for future events")
        i, x = rival_inventory(h, c, focal, self.rival)
        j = (
            c.product_index[self.product]
            if isinstance(self.product, str)
            else self.product
        )
        local = c.products_of[i].index(j)
        return all(int(u) <= n for u, n in zip(c.A[i][:, local], x))


@dataclass(frozen=True)
class LegInventoryAtMost:
    """Threshold event for one rival residual leg inventory (Eqs. 24–25, 30)."""

    rival: str | int
    leg: str | int
    k: int

    def evaluate(self, h, c, focal):
        """Test this declared event against one model hypothesis (Eqs. 24–25, 30)."""
        i, x = rival_inventory(h, c, focal, self.rival)
        local = (
            c.legs_of[i].index(c.leg_index[self.leg])
            if isinstance(self.leg, str)
            else self.leg
        )
        return x[local] <= self.k


@dataclass(frozen=True)
class AllOf:
    """Conjunction of explicitly declared joint events (Eqs. 24–25, 30)."""

    events: tuple

    def evaluate(self, h, c, focal):
        """Test this declared event against one model hypothesis (Eqs. 24–25, 30)."""
        return all(e.evaluate(h, c, focal) for e in self.events)


@dataclass(frozen=True)
class AnyOf:
    """Disjunction of explicitly declared joint events (Eqs. 24–25, 30)."""

    events: tuple

    def evaluate(self, h, c, focal):
        """Test this declared event against one model hypothesis (Eqs. 24–25, 30)."""
        return any(e.evaluate(h, c, focal) for e in self.events)


JointAvailabilityEvent = ItinerarySellable
