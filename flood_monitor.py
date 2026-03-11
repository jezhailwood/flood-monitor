from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px

if TYPE_CHECKING:
    from .api_client import APIClient


@dataclass(frozen=True, slots=True)
class Reading:
    timestamp: datetime
    level: float


class MeasurementStation:
    def __init__(self, station_id: str | int, api_client: APIClient) -> None:
        self._validate_station_id(station_id)

        self.station_id = station_id
        self.api_client = api_client

        self.label: str | None = None
        self.river_name: str | None = None
        self.catchment_name: str | None = None
        self.lon: float | None = None
        self.lat: float | None = None
        self.latest_reading: Reading | None = None
        self.highest_recent: Reading | None = None
        self.max_on_record: Reading | None = None
        self.min_on_record: Reading | None = None
        self.typical_range_high: float | None = None
        self.typical_range_low: float | None = None
        self.state: str | None = None
        self.trend: str | None = None

        self._loaded = False

    def _validate_station_id(self, station_id: str | int) -> None:
        if not isinstance(station_id, (str, int)):
            raise TypeError(
                f"station_id must be str or int, got {type(station_id).__name__}"
            )
        if isinstance(station_id, str):
            if not station_id.strip():
                raise ValueError("station_id must not be empty or blank")
            if station_id != station_id.strip():
                raise ValueError("station_id must not have surrounding whitespace")
        if isinstance(station_id, int) and station_id <= 0:
            raise ValueError("station_id must be a positive integer")

    def load(self) -> None:
        data = self.api_client.get(
            "flood-monitoring",
            "id",
            "stations",
            self.station_id,
        )

        items = data.get("items", {})
        stage_scale = items.get("stageScale", {})
        measures = items.get("measures", {})

        self.label = items.get("label")
        self.river_name = items.get("riverName")
        self.catchment_name = items.get("catchmentName")
        self.lon = items.get("long")
        self.lat = items.get("lat")
        self.latest_reading = self._build_reading(measures.get("latestReading"))
        self.highest_recent = self._build_reading(stage_scale.get("highestRecent"))
        self.max_on_record = self._build_reading(stage_scale.get("maxOnRecord"))
        self.min_on_record = self._build_reading(stage_scale.get("minOnRecord"))
        self.typical_range_high = self._parse_level(stage_scale.get("typicalRangeHigh"))
        self.typical_range_low = self._parse_level(stage_scale.get("typicalRangeLow"))

        self.state = self._get_state()
        self.trend = self._get_trend()

        self._loaded = True

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("MeasurementStation not loaded. Call load() first.")

    def _build_reading(self, data: dict | None) -> Reading | None:
        if data is None:
            return None

        timestamp = self._parse_timestamp(data.get("dateTime"))
        level = self._parse_level(data.get("value"))

        if timestamp is None or level is None:
            return None
        return Reading(timestamp=timestamp, level=level)

    def _parse_timestamp(self, timestamp: str | None) -> datetime | None:
        if timestamp is None:
            return None

        # Convert trailing Z (UTC) to +00:00.
        if timestamp.endswith("Z"):
            timestamp = timestamp[:-1] + "+00:00"

        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None

    def _parse_level(self, level: float | int | None) -> float | None:
        if level is None:
            return None
        return float(level)

    def _get_state(self) -> str:
        latest_reading_level = (
            self.latest_reading.level if self.latest_reading else None
        )

        if (
            latest_reading_level is None
            or self.typical_range_low is None
            or self.typical_range_high is None
        ):
            return "unknown"

        if latest_reading_level <= self.typical_range_low:
            return "low"
        if latest_reading_level >= self.typical_range_high:
            return "high"
        return "normal"

    def _get_trend(self, threshold: float = 1.0, limit: int = 5) -> str:
        readings = self.get_readings(limit=limit)

        if len(readings) < 2:
            return "unknown"

        # Convert cm/h to m/s.
        threshold_ms = threshold / 360_000

        # Express timestamps as elapsed seconds from the first reading.
        t0 = readings[0].timestamp.timestamp()
        x = np.array([r.timestamp.timestamp() - t0 for r in readings])
        y = np.array([r.level for r in readings])

        # slope gives rate of change in m/s.
        slope, _ = np.polyfit(x, y, 1)

        if slope > threshold_ms:
            return "rising"
        if slope < -threshold_ms:
            return "falling"
        return "steady"

    def get_readings(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        days: int | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[Reading]:
        self._validate_readings_args(start=start, end=end, days=days, limit=limit)

        if days is not None:
            now = datetime.now(UTC)
            start = now - timedelta(days=days)
            end = None

        params = self._build_readings_params(start=start, end=end, limit=limit)

        data = self.api_client.get(
            "flood-monitoring",
            "id",
            "stations",
            self.station_id,
            "readings",
            params=params,
        )

        readings = []
        for item in data.get("items", []):
            reading = self._build_reading(item)
            if reading is not None:
                readings.append(reading)

        return sorted(readings, key=attrgetter("timestamp"), reverse=reverse)

    def _validate_readings_args(
        self,
        *,
        start: datetime | None,
        end: datetime | None,
        days: int | None,
        limit: int | None,
    ) -> None:
        if start is not None and not isinstance(start, datetime):
            raise TypeError("start must be a datetime or None")

        if end is not None and not isinstance(end, datetime):
            raise TypeError("end must be a datetime or None")

        if days is not None and not isinstance(days, int):
            raise TypeError("days must be an int or None")

        if limit is not None and not isinstance(limit, int):
            raise TypeError("limit must be an int or None")

        if days is not None and days <= 0:
            raise ValueError("days must be a positive integer")

        if limit is not None and limit <= 0:
            raise ValueError("limit must be a positive integer")

        # If end is provided, start must be provided.
        if end is not None and start is None:
            raise ValueError("end was provided without start")

        # Days is mutually exclusive with start/end.
        if days is not None and (start is not None or end is not None):
            raise ValueError("days is mutually exclusive with start/end")

        if start is not None and end is not None:
            start_aware = start.tzinfo is not None and start.utcoffset() is not None
            end_aware = end.tzinfo is not None and end.utcoffset() is not None
            # Enforce timezone consistency (both naive or both aware).
            if start_aware != end_aware:
                raise ValueError(
                    "start and end must both be timezone-aware or both naive"
                )
            # Avoid mixing different timezones.
            if start_aware and end_aware and start.utcoffset() != end.utcoffset():
                raise ValueError("start and end must use the same timezone offset")
            # Ensure logical ordering.
            if start >= end:
                raise ValueError("start must be < end")

    def _build_readings_params(
        self,
        *,
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
    ) -> dict[str, str]:
        params = {"parameter": "level"}

        if limit is not None:
            params["_limit"] = str(limit)

        # Most recent readings (no date filter).
        if start is None and end is None:
            params["_sorted"] = ""
            return params

        # Readings since `start` (open-ended).
        if end is None:
            params["since"] = start.isoformat()
            return params

        # Readings between `start` and `end` (bounded).
        params["startdate"] = start.date().isoformat()
        params["enddate"] = end.date().isoformat()
        return params

    def plot_map(self) -> None:
        self._require_loaded()

        fig = px.scatter_map(
            lat=[self.lat],
            lon=[self.lon],
            hover_name=[self.label],
        )

        fig.show()

    def plot_chart(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        days: int | None = None,
    ) -> None:
        self._require_loaded()

        readings = self.get_readings(start=start, end=end, days=days)
        timestamps = [r.timestamp for r in readings]
        levels = [r.level for r in readings]

        fig = px.line(
            x=timestamps, y=levels, labels={"x": "Timestamp", "y": "Level (m)"}
        )

        fig.show()
