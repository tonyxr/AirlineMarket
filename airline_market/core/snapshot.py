"""Immutable specification containers and snapshot protocols."""

from collections.abc import Mapping
from typing import Any, Protocol


class Snapshotable(Protocol):
    """
    Snapshot/restore protocol for reproducible counterfactual execution (model §10,
    reproducibility conventions).
    """

    def snapshot(self) -> Any:
        """
        Copy mutable execution/filter state for reproducible branching (model §10,
        reproducibility conventions).
        """
        ...

    def restore(self, snapshot: Any) -> None:
        """
        Restore a previously captured execution/filter snapshot (model §10,
        reproducibility conventions).
        """
        ...


class FrozenMap(Mapping):
    """
    Recursively immutable mapping with safe shared snapshot semantics (model §10,
    reproducibility conventions).
    """

    def __init__(self, values=()):
        """
        Bind declared inputs and initialize FrozenMap bookkeeping (model §10,
        reproducibility conventions).
        """
        self._data = {k: deep_freeze(v) for k, v in dict(values).items()}

    def __getitem__(self, key):
        """
        Return the stored immutable mapping value (model §10, reproducibility
        conventions).
        """
        return self._data[key]

    def __iter__(self):
        """
        Iterate mapping keys in their preserved declaration order (model §10,
        reproducibility conventions).
        """
        return iter(self._data)

    def __len__(self):
        """
        Return the number of stored mapping entries (model §10, reproducibility
        conventions).
        """
        return len(self._data)

    def __deepcopy__(self, memo):
        """
        Share this immutable value safely when branching mutable state (model §10,
        reproducibility conventions).
        """
        return self

    def __reduce__(self):
        """
        Serialize the immutable mapping without hidden environment state (model §10,
        reproducibility conventions).
        """
        return FrozenMap, (dict(self._data),)

    def __repr__(self):
        """
        Return the diagnostic representation of this value (model §10, reproducibility
        conventions).
        """
        return f"FrozenMap({self._data!r})"


def deep_freeze(value: Any) -> Any:
    """
    Recursively freeze mappings, sequences and array-backed specification data (model
    §10, reproducibility conventions).
    """
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (tuple, list)):
        return tuple(deep_freeze(v) for v in value)
    return value
