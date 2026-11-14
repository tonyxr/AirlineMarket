"""Canonical serialization for reproducible specifications and artifacts."""

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any


def jsonable(obj: Any) -> Any:
    """
    Convert immutable specs, arrays and scalar values to JSON-compatible data (model
    §10, reproducibility conventions).
    """
    if dataclasses.is_dataclass(obj):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return obj


def canonical_json(obj: Any) -> str:
    """
    Serialize deterministically with sorted keys and finite numbers (model §10,
    reproducibility conventions).
    """
    return json.dumps(
        jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_text(text: str) -> str:
    """
    Fingerprint UTF-8 text for reproducibility checks (model §10, reproducibility
    conventions).
    """
    return hashlib.sha256(text.encode()).hexdigest()


def spec_hash(spec: Any) -> str:
    """
    Fingerprint the canonical specification or schema (model §10, reproducibility
    conventions).
    """
    return sha256_text(canonical_json(spec))


schema_hash = spec_hash
config_hash = spec_hash
