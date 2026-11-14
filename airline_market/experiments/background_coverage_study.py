"""Isolated national-to-core demand coverage audit on already acquired raw data."""

import json
from pathlib import Path

from ..campaign.artifacts import file_hash
from ..pipeline.national_coverage import audit_national


def audit(cell, context):
    """Run the frozen diagnostic specification without changing any live model code."""
    p = cell["parameters"]
    if file_hash(p["audit_config"]) != p["audit_config_sha256"]:
        raise ValueError("national coverage audit configuration changed")
    return audit_national(json.loads(Path(p["audit_config"]).read_text()), context)
