from dataclasses import dataclass, field
from pathlib import Path
import json
from ..core.errors import DataValidationError


@dataclass(frozen=True)
class MarketScope:
    """
    Dated hub/region/carrier selection, local source paths and build assumptions (model
    §§1–3, calibration assumptions).
    """

    name: str
    builder: str = "public_data"
    hubs: tuple = ()
    carriers: tuple = ()
    departure_date: str = ""
    sources: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name, data):
        """
        Parse declared scope fields and preserve source-specific build parameters (model
        §§1–3, calibration assumptions).
        """
        known = {"builder", "hubs", "carriers", "departure_date", "sources"}
        if data.get("builder", "public_data") == "public_data" and (
            not data.get("hubs") or not data.get("carriers")
        ):
            raise DataValidationError("public-data scope requires hubs and carriers")
        return cls(
            name,
            **{
                k: tuple(v) if k in ("hubs", "carriers") else v
                for k, v in data.items()
                if k in known
            },
            params={k: v for k, v in data.items() if k not in known},
        )


def load_scope(name, catalog="config/market_catalog.json"):
    """
    Load a named scope from the local market catalog (model §§1–3, calibration
    assumptions).
    """
    data = json.loads(Path(catalog).read_text())
    if name not in data["scopes"]:
        raise DataValidationError(f"unknown scope {name}")
    return MarketScope.from_dict(name, data["scopes"][name])
