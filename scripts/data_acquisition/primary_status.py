"""Report only the dependency-reviewed primary data; do not restart acquisition."""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from download import atomic


def main():
    """Join receipts/stream evidence without turning optional archives into requirements."""
    root = Path(__file__).resolve().parents[2]
    requirements = json.loads((root / "config/data/primary_ord_2022.json").read_text())
    ids = [id for source in requirements["requirements"] for id in source["ids"]]
    files = []
    for id in ids:
        receipt = root / "results/data_acquisition/files" / (id + ".json")
        r = (
            json.loads(receipt.read_text())
            if receipt.exists()
            else {"id": id, "status": "not_started"}
        )
        vp = root / "results/data_acquisition/validation" / (id + ".json")
        v = json.loads(vp.read_text()) if vp.exists() else {}
        matched = (
            v.get("sha256") == r.get("sha256")
            and v.get("status") == "raw_stream_verified"
        )
        files.append(
            {
                key: r.get(key)
                for key in ["id", "source", "status", "path", "bytes", "sha256"]
            }
            | {
                "rows": v.get("rows") if matched else r.get("rows"),
                "raw_validation_status": v.get("status")
                if matched
                else "not_stream_verified",
                "raw_validation_path": str(vp.relative_to(root)) if matched else None,
            }
        )
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "raw_primary_acquired"
        if all(r["status"] == "downloaded" for r in files)
        else "raw_primary_acquisition_in_progress",
        "required_files": len(files),
        "downloaded_files": sum(r["status"] == "downloaded" for r in files),
        "stream_verified_csv_files": sum(
            r["raw_validation_status"] == "raw_stream_verified" for r in files
        ),
        "downloaded_primary_bytes": sum(
            r.get("bytes") or 0 for r in files if r["status"] == "downloaded"
        ),
        "files": files,
        "regional_attribution_and_calibration": "pending; raw acquisition is not empirical calibration",
        "broad_queues": "stopped by owner scope correction; must not resume automatically",
        "external_storage_required_for_current_scope": False,
    }
    atomic(root / "results/data_acquisition/primary_model_inventory.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
