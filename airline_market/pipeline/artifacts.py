from pathlib import Path
import json
from ..spec import dump_spec
from ..core.hashing import spec_hash, jsonable


def save_calibration(scope, result, root="data/calibrated"):
    """
    Write the calibrated specification and its fit/provenance artifacts (model §§1–3,
    calibration assumptions).
    """
    out = Path(root) / scope / spec_hash(result.spec)[:16]
    out.mkdir(parents=True, exist_ok=True)
    dump_spec(result.spec, out / "spec.json")
    (out / "fit.json").write_text(
        json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n"
    )
    return out
