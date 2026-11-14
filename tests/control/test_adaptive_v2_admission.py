"""S1/A.15/R2: release independent work under the actual12GiB reservation cap."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "adaptive_v2", Path("scripts/airline_market_s1_adaptive.py")
)
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


def test_large_f1_reservation_blocks_new_cells():
    assert adaptive.available_budget("draining_v2", "stopped_v2") == 0


def test_finishing_f5_allows_five_screen_cells_plus_event_and_f4():
    budget = adaptive.available_budget("stopped_v2", "draining_v2")
    assert budget == 8.5
    admitted = int(budget // 1.5)
    assert admitted == 5 and admitted * 1.5 + 2 + 1 + 0.5 <= 12


def test_f5_release_admits_seven_screen_cells_with_event_and_f4():
    budget = adaptive.available_budget("stopped_v2", "stopped_v2")
    assert budget == 10.5
    admitted = int(budget // 1.5)
    assert admitted == 7 and admitted * 1.5 + 1 + 0.5 <= 12
