from __future__ import annotations
import bisect,re
from dataclasses import dataclass
from datetime import datetime,timezone
from typing import Sequence,Optional,Dict,Tuple,Any,Mapping,List
from ...core.errors import DataValidationError

@dataclass(frozen=True)
class WeatherObservation:
    timestamp_utc: datetime
    weather: str
    temperature_c: Optional[float]
    precipitation_mm: Optional[float]
    wind_speed_mps: Optional[float]
    visibility_m: Optional[float]
    station: str

class NOAAWeatherIndex:

    def __init__(self, observations: Sequence[WeatherObservation], max_gap_minutes: float = 120.0):
        ordered = sorted(observations, key=lambda item: item.timestamp_utc)
        self.observations = tuple(ordered)
        self._timestamps = tuple(item.timestamp_utc.timestamp() for item in ordered)
        self.max_gap_seconds = float(max_gap_minutes) * 60.0

    def nearest(self, local_datetime: datetime) -> Optional[WeatherObservation]:
        if not self.observations:
            return None
        stamp = local_datetime
        if stamp.tzinfo is None:
            raise DataValidationError("weather lookup requires a timezone-aware datetime")
        target = stamp.astimezone(timezone.utc).timestamp()
        index = bisect.bisect_left(self._timestamps, target)
        candidates = []
        if index < len(self.observations):
            candidates.append(self.observations[index])
        if index > 0:
            candidates.append(self.observations[index - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda item: abs(item.timestamp_utc.timestamp() - target))
        gap = abs(best.timestamp_utc.timestamp() - target)
        return best if gap <= self.max_gap_seconds else None

    def category_probabilities(self) -> Dict[str, float]:
        # One record per station-hour prevents irregular sub-hourly reporting
        # from overweighting severe periods.
        hourly: Dict[Tuple[str, int], str] = {}
        for observation in self.observations:
            hour = int(observation.timestamp_utc.timestamp() // 3600)
            hourly[(observation.station, hour)] = observation.weather
        counts = {"clear": 0, "rain": 0, "snow": 0}
        for value in hourly.values():
            counts[value if value in counts else "clear"] += 1
        total = float(sum(counts.values()))
        return {key: value / total for key, value in counts.items()} if total else {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a"}

def _scaled_noaa(value: Any, scale: float, missing: int = 9999) -> Optional[float]:
    if _is_missing(value):
        return None
    token = str(value).split(",", 1)[0].strip()
    try:
        raw = int(token)
    except ValueError:
        return None
    if abs(raw) >= missing:
        return None
    return float(raw) / float(scale)

def _noaa_precipitation(row: Mapping[str, Any]) -> Optional[float]:
    depths: List[float] = []
    for key in ("AA1", "AA2", "AA3", "AA4"):
        value = row.get(key)
        if _is_missing(value):
            continue
        parts = str(value).split(",")
        if len(parts) < 2:
            continue
        try:
            depth = int(parts[1])
        except ValueError:
            continue
        if 0 <= depth < 9999:
            depths.append(float(depth) / 10.0)
    return max(depths) if depths else None

def _noaa_weather(row: Mapping[str, Any], temperature_c: Optional[float], precip_mm: Optional[float]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ("REM", "AW1", "MW1", "MW2")).upper()
    if re.search(r"(^|[ +])-?(SN|SG|PL|GR|GS)([ +]|$)", text) or any(
        code in text for code in (",7", ",85", ",86")
    ):
        return "snow"
    if re.search(r"(^|[ +])-?(RA|DZ|SHRA|TSRA|FZRA)([ +]|$)", text):
        return "snow" if temperature_c is not None and temperature_c <= 0.5 else "rain"
    if precip_mm is not None and precip_mm > 0.0:
        return "snow" if temperature_c is not None and temperature_c <= 0.5 else "rain"
    return "clear"
