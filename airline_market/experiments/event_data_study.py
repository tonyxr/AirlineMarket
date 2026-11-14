"""A.15/V10: exact scoped extraction of acquired predeclared event sources."""

import time

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.pipeline.events import spirit_candidate_scope
from airline_market.pipeline.sources.db1c import scope_market_archive


def run(cell, context):
    """Compile the pre-event scope and all seven DB1C months without fitting."""
    p = cell["parameters"]
    current = {
        "stage": "source_identity",
        "files_completed": 0,
        "files_total": len(p["db1c_sources"]),
    }
    context.counters = lambda: dict(current)
    for source in [*p["pre_event_marketing_sources"], *p["db1c_sources"]]:
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("event source changed: " + source["path"])
    current.update(stage="pre_event_market_scope")
    scope = spirit_candidate_scope(
        [(s["path"], (s["year"], s["month"])) for s in p["pre_event_marketing_sources"]]
    )
    atomic_json(context.directory / "E3_candidate_scope.json", scope)
    if not scope["directed_markets"]:
        raise ValueError("no eligible pre-event marketed-service pairs")
    summaries = []
    started = time.perf_counter()
    for source in p["db1c_sources"]:
        current.update(
            stage="scoped_db1c",
            reporting_month=source["month"],
            rows_read=0,
            rows_kept=0,
        )

        def progress(read, kept):
            current.update(rows_read=read, rows_kept=kept)

        destination = context.directory / f"db1c_2026_{source['month']:02d}.parquet"
        summary = scope_market_archive(
            source["path"],
            scope["directed_markets"],
            destination,
            expected_reporting_period=(source["year"], source["month"]),
            sample_expansion=p["sample_expansion"],
            batch_rows=p["batch_rows"],
            progress=progress,
        )
        summary.update(source_path=source["path"], source_sha256=source["sha256"])
        atomic_json(
            context.directory / f"db1c_2026_{source['month']:02d}.json", summary
        )
        summaries.append(summary)
        current["files_completed"] += 1
    report = {
        "plan_id": "A.15/V10",
        "status": "scoped_stream_complete",
        "months": len(summaries),
        "airport_pairs": len(scope["airport_pairs"]),
        "directed_markets": len(scope["directed_markets"]),
        "rows_read": sum(s["rows_read"] for s in summaries),
        "rows_kept": sum(s["rows_kept"] for s in summaries),
        "reporting_period_mismatch_rows": sum(
            s["reporting_period_mismatch_rows"] for s in summaries
        ),
        "all_selected_member_crc_verified": all(
            s["member_crc_verified"] for s in summaries
        ),
        "months_summary": summaries,
        "scoped_stream_wall_seconds": time.perf_counter() - started,
        "full_V10_pass": False,
        "scientific_budget": None,
        "scope": "Observed pre-event marketed-service candidate; no final event network, fit, legal dated itinerary, hidden inventory or net-profit identification.",
    }
    atomic_json(context.directory / "stream_summary.json", report)
    return {
        **{k: v for k, v in report.items() if k not in ("months_summary",)},
        "status": "completed",
        "summary_path": "stream_summary.json",
        "simulated_physical_cohorts": 0,
        "fits": 0,
    }
