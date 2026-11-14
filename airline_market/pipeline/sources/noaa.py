from .common import stream


def load_weather(scope, **kwargs):
    """
    Read the station-scoped weather subset with source provenance (model §§1–3,
    calibration assumptions).
    """
    return stream(scope, "noaa", ("station", "date"), **kwargs)


class AirportWeatherIndex:
    """
    Airport-to-station daily weather lookup (model §§1–3, calibration assumptions).
    """

    def __init__(self, frame, stations):
        """
        Bind declared inputs and initialize AirportWeatherIndex bookkeeping (model
        §§1–3, calibration assumptions).
        """
        self.rows = {
            (str(r.station), str(r.date)[:10]): r._asdict() for r in frame.itertuples()
        }
        self.stations = stations

    def at(self, airport, date):
        """
        Look up the declared airport/station record for a date (model §§1–3, calibration
        assumptions).
        """
        return self.rows.get((self.stations[airport], str(date)[:10]), {})


from ...vendor.ride_response.weather import NOAAWeatherIndex, WeatherObservation


def hourly_indexes(scope, **kwargs):
    """
    Build airport station indexes from decoded daily or Global Hourly records (model
    §§1–3, calibration assumptions).
    """
    from datetime import datetime, timezone

    f = load_weather(scope, **kwargs)
    result = {}
    for airport, station in scope.params.get("weather_stations", {}).items():
        observations = []
        for row in f[f.station == station].itertuples():
            stamp = datetime.fromisoformat(str(row.date).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            raw = row._asdict()
            if "TMP" in raw or "AA1" in raw:
                from ...vendor.ride_response.weather import (
                    _scaled_noaa,
                    _noaa_precipitation,
                    _noaa_weather,
                )

                temp = _scaled_noaa(raw.get("TMP"), 10.0)
                p = _noaa_precipitation(raw)
                weather = _noaa_weather(raw, temp, p)
                wind = str(raw.get("WND", "")).split(",")
                wind = _scaled_noaa(wind[3], 10.0) if len(wind) >= 4 else None
                visibility = _scaled_noaa(raw.get("VIS"), 1.0, missing=999999)
            else:
                import math

                def numeric(value):
                    """
                    Convert a finite weather measurement or retain explicit missingness
                    (model §§1–3, calibration assumptions).
                    """
                    try:
                        return float(value) if math.isfinite(float(value)) else None
                    except (TypeError, ValueError):
                        return None

                p = numeric(getattr(row, "precipitation", None))
                snow = numeric(getattr(row, "snow", None))
                temp = numeric(getattr(row, "temperature", None))
                wind = visibility = None
                weather = (
                    "snow" if snow and snow > 0 else "rain" if p and p > 0 else "clear"
                )
            observations.append(
                WeatherObservation(
                    timestamp_utc=stamp,
                    weather=weather,
                    temperature_c=temp,
                    precipitation_mm=p,
                    wind_speed_mps=wind,
                    visibility_m=visibility,
                    station=station,
                )
            )
        result[airport] = NOAAWeatherIndex(observations)
    return result
