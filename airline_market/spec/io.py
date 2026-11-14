"""JSON specification IO and explicit overrides; no simulator/data coupling."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import types
from typing import Any, Union, get_args, get_origin, get_type_hints
from ..core.errors import SpecError
from ..core.hashing import jsonable
from .model import MarketSpec
from .schedule import PeriodSchedule


def _decode(tp, value, path: str):
    """Recursively decode typed immutable specification fields (Eqs. 1–19)."""
    if tp is Any:
        return value
    if value is None:
        return None
    if tp is PeriodSchedule:
        try:
            return PeriodSchedule.parse(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecError(path, str(exc)) from exc
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, types.UnionType):
        candidates = [x for x in args if x is not type(None)]
        if (
            PeriodSchedule in candidates
            and isinstance(value, Mapping)
            and any(k in value for k in ("constant", "piecewise", "breakpoints"))
        ):
            return _decode(PeriodSchedule, value, path)
        for candidate in candidates:
            try:
                return _decode(candidate, value, path)
            except (SpecError, TypeError, ValueError, KeyError):
                continue
        raise SpecError(path, "value does not match the declared type")
    if is_dataclass(tp):
        hints = get_type_hints(tp)
        known = {f.name for f in fields(tp)}
        extra = set(value) - known
        if extra:
            raise SpecError(path, f"unknown fields: {sorted(extra)}")
        try:
            return tp(
                **{k: _decode(hints[k], v, f"{path}.{k}") for k, v in value.items()}
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SpecError):
                raise
            raise SpecError(path, str(exc)) from exc
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise SpecError(path, "expected a sequence")
        if not (len(args) == 2 and args[1] is Ellipsis) and len(value) != len(args):
            raise SpecError(path, "incorrect tuple length")
        return tuple(
            _decode(
                args[0] if len(args) == 2 and args[1] is Ellipsis else args[i],
                v,
                f"{path}[{i}]",
            )
            for i, v in enumerate(value)
        )
    if origin in (dict, Mapping):
        return {
            args[0](k): _decode(args[1], v, f"{path}.{k}") for k, v in value.items()
        }
    return value


def load_spec(source: str | Path | Mapping, *, validate: bool = True) -> MarketSpec:
    """
    Read, decode and validate a market or compiled-market JSON envelope (Eqs. 1–19).
    """
    data = (
        dict(source)
        if isinstance(source, Mapping)
        else json.loads(Path(source).read_text())
    )
    if data.get("schema") == "airline-compiled-market/1":
        data = data["spec"]
    spec = _decode(MarketSpec, data, "$")
    if validate:
        from .validate import validate_spec

        validate_spec(spec)
    return spec


def dump_spec(spec: MarketSpec, path: str | Path | None = None) -> str:
    """Serialize an immutable specification as finite, reproducible JSON (Eqs. 1–19)."""
    text = json.dumps(jsonable(spec), indent=2, allow_nan=False) + "\n"
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


def apply_overrides(spec: MarketSpec, overrides: Mapping[str, Any]) -> MarketSpec:
    """
    Apply validated aliases or explicit dotted specification overrides (Eqs. 1–19).
    """
    data = jsonable(spec)
    # Convenience aliases encode every original Config field; generic dotted
    # paths continue to operate on any market specification.
    aliases = {k: v for k, v in overrides.items() if "." not in k}
    if aliases:
        from ..legacy_toy.model import Config
        from .library_builders import toy_hub

        allowed = set(Config.__dataclass_fields__)
        if spec.name != "toy_hub" or set(aliases) - allowed:
            raise SpecError(
                "$.overrides", f"unknown aliases: {sorted(set(aliases) - allowed)}"
            )
        settings = dict(spec.metadata["legacy_config"])
        settings.update(aliases)
        for key in ("own_capacity", "rival_hd_capacities"):
            settings[key] = tuple(settings[key])
        spec = toy_hub(Config(**settings))
        data = jsonable(spec)
    for path, value in overrides.items():
        if "." not in path:
            continue
        parts = path.split(".")
        current = data
        try:
            for key in parts[:-1]:
                current = (
                    current[int(key)] if isinstance(current, list) else current[key]
                )
            key = int(parts[-1]) if isinstance(current, list) else parts[-1]
            if not isinstance(current, list) and key not in current:
                raise KeyError(key)
            current[key] = value
        except (KeyError, IndexError, TypeError) as exc:
            raise SpecError("$.overrides." + path, "path does not exist") from exc
    return load_spec(data)
