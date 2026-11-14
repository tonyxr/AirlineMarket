from __future__ import annotations
import copy, math
from dataclasses import dataclass,field
from typing import Any,Dict,List,Sequence,Tuple,Optional,Mapping
from scipy.special import expit
import numpy as np
import torch
class NumericalPolicyActionError(FloatingPointError):
    """A finite policy coordinate could not be mapped to a finite action."""

    def __init__(self, coordinates: np.ndarray, message: str) -> None:
        super().__init__(message)
        self.coordinates = np.asarray(coordinates, dtype=np.float64).copy()

@dataclass(frozen=True)
class ActionTransform:
    """Mapping between policy coordinates and physical market controls.

    ``kinds`` entries are:

    ``"log_positive"``
        Coordinate ``z`` maps to ``exp(z)``.  It is positive and uncapped.
    ``"identity"``
        Coordinate and physical action are identical and unbounded.
    ``"bounded"``
        A coefficient action mapped through sigmoid to finite ``low/high``.
    """

    names: Tuple[str, ...]
    kinds: Tuple[str, ...]
    low: Tuple[float, ...] = ()
    high: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.kinds):
            raise ValueError("names and kinds must have the same non-zero length")
        allowed = {"log_positive", "identity", "bounded"}
        unknown = set(self.kinds) - allowed
        if unknown:
            raise ValueError(f"unknown action transform kinds: {sorted(unknown)}")
        if any(kind == "bounded" for kind in self.kinds):
            if len(self.low) != self.dim or len(self.high) != self.dim:
                raise ValueError("bounded transforms require low/high for every dimension")
            for index, kind in enumerate(self.kinds):
                if kind == "bounded" and not self.low[index] < self.high[index]:
                    raise ValueError(f"invalid bounds for action {self.names[index]!r}")

    @property
    def dim(self) -> int:
        return len(self.names)

    def coordinates_to_physical_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.dim:
            raise ValueError(f"expected action width {self.dim}, got {coordinates.shape[-1]}")
        columns: List[torch.Tensor] = []
        for index, kind in enumerate(self.kinds):
            value = coordinates[..., index]
            if kind == "log_positive":
                value = torch.exp(value)
            elif kind == "bounded":
                lo = float(self.low[index])
                hi = float(self.high[index])
                value = lo + torch.sigmoid(value) * (hi - lo)
            columns.append(value)
        return torch.stack(columns, dim=-1)

    def coordinates_to_physical(self, coordinates: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(coordinates, dtype=np.float64)
        if array.shape[-1] != self.dim:
            raise ValueError(f"expected action width {self.dim}, got {array.shape[-1]}")
        if not np.all(np.isfinite(array)):
            raise NumericalPolicyActionError(
                array, "policy emitted a non-finite native action coordinate"
            )
        result = array.copy()
        for index, kind in enumerate(self.kinds):
            if kind == "log_positive":
                # No economic cap.  Overflow is detected below instead of
                # silently replacing the learner's action with a guardrail.
                with np.errstate(over="ignore", invalid="ignore"):
                    result[..., index] = np.exp(array[..., index])
            elif kind == "bounded":
                lo = float(self.low[index])
                hi = float(self.high[index])
                result[..., index] = lo + expit(array[..., index]) * (hi - lo)
        invalid_positive = any(
            kind == "log_positive" and np.any(result[..., index] <= 0.0)
            for index, kind in enumerate(self.kinds)
        )
        if not np.all(np.isfinite(result)) or invalid_positive:
            # exp(z) may underflow to the finite value 0.0 as well as overflow
            # to infinity.  Both are numerical failures of an otherwise
            # uncapped transform; neither is clipped or replaced.
            raise NumericalPolicyActionError(
                array,
                "policy coordinate overflowed or underflowed its physical action",
            )
        return result

    def physical_to_coordinates(self, physical: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(physical, dtype=np.float64)
        if array.shape[-1] != self.dim:
            raise ValueError(f"expected action width {self.dim}, got {array.shape[-1]}")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError("cannot store a non-finite physical action")
        result = array.copy()
        for index, kind in enumerate(self.kinds):
            if kind == "log_positive":
                if np.any(array[..., index] <= 0.0):
                    raise ValueError(f"action {self.names[index]!r} must be positive")
                result[..., index] = np.log(array[..., index])
            elif kind == "bounded":
                lo = float(self.low[index])
                hi = float(self.high[index])
                unit = np.clip((array[..., index] - lo) / (hi - lo),1e-6,1-1e-6)
                result[..., index] = np.log(unit)-np.log1p(-unit)
        return result

@dataclass
class Episode:
    """One replay episode with ``T + 1`` observations and ``T`` transitions."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        self.dones = np.asarray(self.dones, dtype=np.float32).reshape(-1)
        self.metadata = copy.deepcopy(dict(self.metadata or {}))
        transitions = self.actions.shape[0]
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("observations and actions must be rank-2 arrays")
        if self.observations.shape[0] != transitions + 1:
            raise ValueError("an episode needs T+1 observations for T actions")
        if self.rewards.shape[0] != transitions or self.dones.shape[0] != transitions:
            raise ValueError("actions, rewards, and dones must have the same length")
        if transitions == 0:
            raise ValueError("empty replay episodes are not supported")
        for name, value in (
            ("observations", self.observations),
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("dones", self.dones),
        ):
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"episode {name} contains non-finite values")

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

class EpisodeBuilder:
    """Small helper that makes transition alignment explicit."""

    def __init__(self, initial_observation: np.ndarray | Sequence[float]) -> None:
        self._observations = [np.asarray(initial_observation, dtype=np.float32).copy()]
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[float] = []

    def append(
        self,
        action: np.ndarray | Sequence[float],
        reward: float,
        next_observation: np.ndarray | Sequence[float],
        done: bool,
    ) -> None:
        if self._dones and self._dones[-1] >= 1.0:
            raise RuntimeError("cannot append after a terminal transition")
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        self._rewards.append(float(reward))
        self._dones.append(float(bool(done)))
        self._observations.append(np.asarray(next_observation, dtype=np.float32).copy())

    def finish(self) -> Episode:
        return Episode(
            observations=np.stack(self._observations),
            actions=np.stack(self._actions),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.float32),
        )

class RunningMeanStd:
    """Streaming training-only observation normalization.

    Categorical/bounded public indicators are represented in their natural
    units.  Standardising an unobserved binary category with the RMS epsilon
    can otherwise turn ``1`` into a four-thousand-sigma input.  The identity
    mask is part of the serialized contract so a checkpoint cannot silently
    change observation coordinates when it is loaded or rolled back.
    """

    def __init__(
        self,
        shape: int,
        epsilon: float = 1e-4,
        identity_indices: Sequence[int] = (),
    ) -> None:
        self.mean = np.zeros(int(shape), dtype=np.float64)
        self.var = np.ones(int(shape), dtype=np.float64)
        self.count = float(epsilon)
        indices = tuple(int(index) for index in identity_indices)
        if len(indices) != len(set(indices)):
            raise ValueError("observation identity indices must be unique")
        if any(index < 0 or index >= int(shape) for index in indices):
            raise ValueError("observation identity index is out of bounds")
        self.identity_mask = np.zeros(int(shape), dtype=bool)
        self.identity_mask[list(indices)] = True

    def update(self, observations: np.ndarray) -> None:
        array = np.asarray(observations, dtype=np.float64).reshape(-1, self.mean.size)
        if array.size == 0:
            return
        batch_mean = array.mean(axis=0)
        batch_var = array.var(axis=0)
        batch_count = float(array.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.var = np.maximum(m2 / total, 1e-8)
        # Identity features never inherit accidental training-distribution
        # moments.  Keeping their stored moments canonical also makes state
        # fingerprints and rollback checks unambiguous.
        self.mean[self.identity_mask] = 0.0
        self.var[self.identity_mask] = 1.0
        self.count = total

    def normalize_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=observations.dtype, device=observations.device)
        std = torch.as_tensor(np.sqrt(self.var), dtype=observations.dtype, device=observations.device)
        normalized = (observations - mean) / std.clamp_min(1e-4)
        if bool(np.any(self.identity_mask)):
            identity = torch.as_tensor(
                self.identity_mask,
                dtype=torch.bool,
                device=observations.device,
            )
            normalized = torch.where(identity, observations, normalized)
        return normalized

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "identity_mask": self.identity_mask.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64).copy()
        var = np.asarray(state["var"], dtype=np.float64).copy()
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError("observation normalizer shape does not match agent")
        stored_mask = np.asarray(
            state.get("identity_mask", np.zeros_like(mean, dtype=bool)),
            dtype=bool,
        )
        if stored_mask.shape != self.identity_mask.shape or not np.array_equal(
            stored_mask, self.identity_mask
        ):
            raise ValueError(
                "observation normalizer identity mask does not match checkpoint"
            )
        self.mean = mean
        self.var = var
        self.count = float(state["count"])
