"""Piecewise-constant integer-period schedules (Eq. 5)."""

from dataclasses import dataclass
from typing import Any
from ..core.snapshot import deep_freeze


@dataclass(frozen=True)
class PeriodSchedule:
    """Right-continuous piecewise-constant values over booking periods (Eqs. 1–19)."""

    breakpoints: tuple[int, ...]
    values: tuple[Any, ...]

    def __post_init__(self):
        """
        Canonicalize declared fields and check the object's construction invariants
        (Eqs. 1–19).
        """
        object.__setattr__(self, "breakpoints", tuple(self.breakpoints))
        object.__setattr__(self, "values", deep_freeze(self.values))
        if (
            not self.breakpoints
            or self.breakpoints[0] != 0
            or len(self.breakpoints) != len(self.values)
        ):
            raise ValueError("schedule starts at 0 and needs one value per breakpoint")
        if any(a >= b for a, b in zip(self.breakpoints, self.breakpoints[1:])):
            raise ValueError("schedule breakpoints must increase strictly")

    def value_at(self, t: int) -> Any:
        """
        Return the active piecewise-constant value at a booking period (Eqs. 1–19).
        """
        if t < 0:
            raise ValueError("negative schedule period")
        for index in range(len(self.breakpoints) - 1, -1, -1):
            if t >= self.breakpoints[index]:
                return self.values[index]

    def as_array(self, T: int):
        """Expand a finite period schedule into ordered values (Eqs. 1–19)."""
        import numpy as np

        return np.asarray([self.value_at(t) for t in range(T)])

    @classmethod
    def parse(cls, value: Any):
        """Parse constant, piecewise or explicit schedule data (Eqs. 1–19)."""
        if isinstance(value, cls):
            return value
        if "constant" in value:
            return cls((0,), (value["constant"],))
        if "piecewise" in value:
            return cls(
                tuple(int(p[0]) for p in value["piecewise"]),
                tuple(p[1] for p in value["piecewise"]),
            )
        return cls(tuple(value["breakpoints"]), tuple(value["values"]))
