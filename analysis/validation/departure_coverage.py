"""Keep marketing coverage while applying observed reporting diversion corrections.

This is a diagnostic only. A regional operator absent from Reporting On-Time
is retained from Marketing On-Time, with its actual path correction unknown.
"""

from pathlib import Path
import argparse
import json

import pandas as pd

from airline_market.pipeline.scoped_io import atomic_json, sha256


def run(source, output):
    """Publish an auditable decomposition without replacing the primary tables."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source / "route_month_reconciliation.csv")
    frame["observed_diversion_segment_adjustment"] = (
        frame.reconstructed_flown_segments - frame.reporting_performed
    )
    frame["marketing_with_observed_adjustment"] = (
        frame.marketing_performed + frame.observed_diversion_segment_adjustment
    )
    frame["adjusted_minus_t100"] = (
        frame.marketing_with_observed_adjustment - frame.departures_performed
    )
    frame["absolute_adjusted_relative_difference"] = (
        frame.adjusted_minus_t100.abs() / frame.departures_performed.clip(lower=1)
    )
    frame.to_csv(output / "marketing_coverage_diagnostic.csv", index=False)
    gap = frame.groupby("operating_carrier", dropna=False)[
        [
            "marketing_scheduled",
            "reporting_scheduled",
            "marketing_performed",
            "reporting_performed",
        ]
    ].sum()
    gap["scheduled_coverage_gap"] = gap.marketing_scheduled - gap.reporting_scheduled
    gap["performed_coverage_gap"] = gap.marketing_performed - gap.reporting_performed
    gap.to_csv(output / "operator_reporting_coverage.csv")
    residual = frame[frame.adjusted_minus_t100 != 0]
    residual.to_csv(output / "unresolved_route_months.csv", index=False)
    summary = {
        "status": "diagnostic_complete_with_unresolved_differences",
        "input_sha256": sha256(source / "route_month_reconciliation.csv"),
        "implementation_sha256": sha256(__file__),
        "scheduled_coverage_gaps_by_operator": gap.loc[
            gap.scheduled_coverage_gap != 0, "scheduled_coverage_gap"
        ].to_dict(),
        "marketing_performed": float(frame.marketing_performed.sum()),
        "t100_departures_performed": float(frame.departures_performed.sum()),
        "marketing_with_observed_diversion_adjustment": float(
            frame.marketing_with_observed_adjustment.sum()
        ),
        "unresolved_cells": len(residual),
        "unresolved_cells_above_2_percent": int(
            (frame.absolute_adjusted_relative_difference > 0.02).sum()
        ),
        "sum_absolute_unresolved_departures": float(
            frame.adjusted_minus_t100.abs().sum()
        ),
        "rule": "Marketing performed + (reconstructed reporting flown hops - reporting performed). Operators absent from the reporting source retain their marketing records; their unobserved diversion corrections are not imputed.",
        "V1_status": "not_yet_passed; no reconciliation tolerance relaxed",
        "compiled_model_status": "not_yet_built",
    }
    atomic_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
