from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple, Dict
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

def paired_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    left_values = np.asarray(left["values"], dtype=np.float64)
    right_values = np.asarray(right["values"], dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("paired evaluations have different seed counts")
    estimate = PairedEstimate.from_values(left_values - right_values)
    return {
        "mean": estimate.mean,
        "standard_error": estimate.standard_error,
        "median": estimate.median,
        "values": list(estimate.values),
        "t": (
            estimate.mean / estimate.standard_error
            if np.isfinite(estimate.standard_error) and estimate.standard_error > 0.0
            else float("nan")
        ),
    }

class FixedPolicy:
    def __init__(self, action: Sequence[float]) -> None:
        self.action = np.asarray(action, dtype=np.float64)

    def __call__(self, observation: np.ndarray, day: int) -> np.ndarray:
        del observation, day
        return self.action.copy()
