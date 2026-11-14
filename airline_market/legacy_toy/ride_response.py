"""Small verbatim helpers vendored from RideResponse, avoiding its SAC import.

Origin: /Users/huali/Desktop/RideResponse/continuous_benchmarks.py
The source project is unchanged. See REUSE.md for scope and attribution.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np

@dataclass(frozen=True)
class PairedEstimate:
    mean: float
    standard_error: float
    median: float
    values: Tuple[float, ...]

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "PairedEstimate":
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            raise ValueError("cannot summarize an empty sample")
        standard_error = (
            float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")
        )
        return cls(
            mean=float(array.mean()),
            standard_error=standard_error,
            median=float(np.median(array)),
            values=tuple(float(value) for value in array),
        )

class FixedPolicy:
    def __init__(self, action: Sequence[float]) -> None:
        self.action = np.asarray(action, dtype=np.float64)

    def __call__(self, observation: np.ndarray, day: int) -> np.ndarray:
        del observation, day
        return self.action.copy()
