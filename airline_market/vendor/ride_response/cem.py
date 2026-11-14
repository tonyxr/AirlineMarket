from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any
import numpy as np


@dataclass
class CEMConfig:
    population: int = 64
    elite_fraction: float = 0.15
    iterations: int = 6
    initial_std: float = 0.75
    minimum_std: float = 0.03
    smoothing: float = 0.20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError("CEM population must be at least two")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must lie in (0, 1]")
        if self.iterations <= 0 or self.initial_std <= 0.0 or self.minimum_std <= 0.0:
            raise ValueError("CEM iterations and standard deviations must be positive")

class CEMOptimizer:
    """Cross-entropy optimizer in policy-coordinate space."""

    def __init__(self, dimension: int, config: Optional[CEMConfig] = None) -> None:
        self.dimension = int(dimension)
        self.config = config or CEMConfig()
        self.rng = np.random.default_rng(int(self.config.seed))

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        *,
        initial_mean: Optional[Sequence[float]] = None,
        initial_std: Optional[Sequence[float] | float] = None,
    ) -> Dict[str, Any]:
        mean = np.zeros(self.dimension, dtype=np.float64)
        if initial_mean is not None:
            mean = np.asarray(initial_mean, dtype=np.float64).reshape(self.dimension)
        if initial_std is None:
            std = np.full(self.dimension, float(self.config.initial_std), dtype=np.float64)
        else:
            std = np.broadcast_to(np.asarray(initial_std, dtype=np.float64), (self.dimension,)).copy()
        # A singleton "elite set" has zero empirical variance.  With the
        # small populations used by smoke diagnostics that immediately
        # collapses CEM around one lucky draw and makes both warm starts and
        # gates optimizer-luck driven.  Two samples are the minimum needed to
        # estimate dispersion; ceil also avoids rounding a positive fraction
        # down prematurely.
        elite_count = min(
            int(self.config.population),
            max(
                2,
                int(np.ceil(self.config.population * self.config.elite_fraction)),
            ),
        )
        best_coordinates = mean.copy()
        best_score = float("-inf")
        history = []
        for iteration in range(int(self.config.iterations)):
            candidates = self.rng.normal(
                loc=mean, scale=std, size=(int(self.config.population), self.dimension)
            )
            # Always re-evaluate the current center.  This makes small-budget
            # runs less sensitive to one unlucky population draw.
            candidates[0] = mean
            scores = np.asarray([float(objective(candidate)) for candidate in candidates])
            if not np.all(np.isfinite(scores)):
                raise FloatingPointError("CEM objective returned a non-finite score")
            order = np.argsort(scores)
            elites = candidates[order[-elite_count:]]
            elite_scores = scores[order[-elite_count:]]
            candidate_mean = elites.mean(axis=0)
            candidate_std = elites.std(axis=0)
            smooth = float(np.clip(self.config.smoothing, 0.0, 1.0))
            mean = smooth * mean + (1.0 - smooth) * candidate_mean
            std = np.maximum(
                smooth * std + (1.0 - smooth) * candidate_std,
                float(self.config.minimum_std),
            )
            index = int(np.argmax(scores))
            if float(scores[index]) > best_score:
                best_score = float(scores[index])
                best_coordinates = candidates[index].copy()
            history.append(
                {
                    "iteration": iteration,
                    "elite_count": int(elite_count),
                    "best_score": float(scores[index]),
                    "elite_mean_score": float(elite_scores.mean()),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                }
            )
        return {
            "coordinates": best_coordinates,
            "score": best_score,
            "elite_count": int(elite_count),
            "final_mean": mean,
            "final_std": std,
            "history": history,
        }
