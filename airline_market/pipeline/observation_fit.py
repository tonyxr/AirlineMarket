from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObservationModelFit:
    """
    Quote-panel summaries and the resulting declared observation defaults (model §§1–3,
    calibration assumptions).
    """

    seats_cap: int
    cap_verified: bool
    seats_distribution: dict
    fare_bin: float
    median_search_gap_days: float
    price_change_by_days: dict

    def shopping_defaults(self):
        """
        Translate fitted panel summaries into explicit shopping-request settings (model
        §§1–3, calibration assumptions).
        """
        return {
            "seats_cap": self.seats_cap,
            "availability": "seats_capped",
            "fare_bin": self.fare_bin,
            "lag_periods": max(1, int(round(self.median_search_gap_days))),
        }


def fit_observation(panel):
    """
    Estimate quote-panel summaries and declared shopping-channel defaults (model §§1–3,
    calibration assumptions).
    """
    import pandas as pd

    group = ["flightDate", "legId"] + (
        ["fareBasisCode"] if "fareBasisCode" in panel else []
    )
    f = panel.sort_values(group + ["searchDate"]).copy()
    previous = f.groupby(group).totalFare.shift()
    gaps = f.groupby(group).searchDate.transform(
        lambda x: pd.to_datetime(x).diff().dt.days
    )
    f["changed"] = (f.totalFare != previous).where(previous.notna())
    changes = f.groupby("days_before_departure").changed.mean().dropna().to_dict()
    seats = f.seatsRemaining.dropna()
    cap = int(seats.max()) if len(seats) else 9
    distribution = {
        str(k): float(v) for k, v in seats.value_counts(normalize=True).items()
    }
    unique = np.sort(f.totalFare.dropna().unique())
    diff = np.diff(unique)
    positive = diff[diff > 0.005]
    fare_bin = float(np.min(positive)) if len(positive) else 0.0
    return ObservationModelFit(
        cap,
        cap == 9,
        distribution,
        fare_bin,
        float(gaps.dropna().median()) if gaps.notna().any() else 1.0,
        {str(k): float(v) for k, v in changes.items()},
    )
