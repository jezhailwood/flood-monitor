"""River-level monitoring using the Environment Agency real time flood-monitoring API.

This module provides `MeasurementStation`, which fetches and encapsulates metadata and
river-level readings for a single measurement station. Readings can be retrieved as
structured data or visualised directly as an interactive map or time-series chart.

Use `MeasurementStation.from_api` to construct a populated instance:

    from api_client import APIClient
    from flood_monitor import MeasurementStation

    station = MeasurementStation.from_api(
        2642, APIClient("https://environment.data.gov.uk")
    )
    station.plot_map()
    station.plot_chart(days=5)
"""

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
    """A single river-level reading from a measurement station.

    Attributes:
        timestamp: The UTC datetime at which the reading was recorded.
        level: The measured river level in metres.
    """

    timestamp: datetime
    level: float


class MeasurementStation:
    """A river-level measurement station.

    Holds station metadata, the latest river-level reading, historical extremes,
    and the typical level range. Use the `from_api` class method to construct a fully
    populated instance.

    Attributes:
        station_id: The unique station identifier used in API requests.
        api_client: The HTTP client used to query the API.
        label: Human-readable station name, if available.
        river_name: Name of the river associated with this station, if available.
        catchment_name: Name of the river catchment, if available.
        lon: Longitude of the station in decimal degrees, if available.
        lat: Latitude of the station in decimal degrees, if available.
        latest_reading: The most recent river-level reading, if available.
        highest_recent: The highest recent river-level reading, if available.
        max_on_record: The highest recorded reading, if available.
        min_on_record: The lowest recorded reading, if available.
        typical_range_high: The top of the typical range band in metres,
            if available.
        typical_range_low: The bottom of the typical range band in metres,
            if available.
        state: Current level state; one of `"low"`, `"normal"`, `"high"`,
            or `"unknown"`.
    """

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
        self._trend: str | None = None

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

    @classmethod
    def from_api(
        cls, station_id: str | int, api_client: APIClient
    ) -> MeasurementStation:
        """Fetch station metadata from the API and return a populated instance.

        This is the intended constructor. It creates a `MeasurementStation`, immediately
        fetches its metadata from the API and returns the populated object.

        Args:
            station_id: The station identifier. May be a non-empty, unpadded string
                or a positive integer.
            api_client: An `APIClient` configured with the API base URL,
                eg `APIClient("https://environment.data.gov.uk")`.

        Returns:
            A `MeasurementStation` populated with metadata from the API.

        Raises:
            TypeError: If `station_id` is not a `str` or `int`.
            ValueError: If `station_id` is blank, has surrounding whitespace, or is a
                non-positive integer.
            requests.HTTPError: If the response status code indicates an error.
            requests.RequestException: If a network-level error occurs.
        """
        station = cls(station_id, api_client)
        station._load()
        return station

    @property
    def trend(self) -> str:
        """The recent river-level trend, computed lazily from the latest readings.

        Fetches the most recent readings and fits a linear regression to estimate the
        rate of change. The result is cached after the first access.

        Returns:
            One of `"rising"`, `"falling"`, `"steady"`, or `"unknown"` if fewer than
            two readings are available.
        """
        if self._trend is None:
            self._trend = self._compute_trend()
        return self._trend

    def _compute_trend(self, threshold: float = 1.0, limit: int = 5) -> str:
        """Compute the current river-level trend from recent readings.

        Fits a least-squares linear regression to the `limit` most recent readings and
        classifies the slope against `threshold`.

        Args:
            threshold: Rate-of-change threshold in cm/h. Slopes with an absolute rate of
                change above this are classified as `"rising"` or `"falling"`;
                those within it are `"steady"`. Defaults to `1.0`cm/h.
            limit: Number of recent readings to include in the regression.
                Defaults to `5`.
        """
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

    def _load(self) -> None:
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

        self.state = self._compute_state()

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

    def _compute_state(self) -> str:
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

    def get_readings(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        days: int | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[Reading]:
        """Fetch river-level readings for this station from the API.

        By default, returns the most recent available readings in chronological order.
        Supply `days` for a rolling window, or `start`/`end` for an explicit date range.

        Args:
            start: Inclusive start of the time range. Required if `end` is given.
            end: Inclusive end of the time range. Must be later than `start`.
                If omitted, the range is open-ended from `start`.
            days: Number of days of history to fetch, counting back from now.
                Mutually exclusive with `start` and `end`.
            limit: Maximum number of readings to return.
            reverse: If `True`, return readings in reverse chronological order.
                Defaults to `False` (chronological).

        Returns:
            A list of `Reading` objects sorted by timestamp, empty if no readings match.

        Raises:
            TypeError: If any argument is the wrong type.
            ValueError: If argument values or combinations are invalid (eg `days` used
                alongside `start`/`end`, `end` without `start`, mismatched timezone
                awareness, or `start >= end`).
            requests.HTTPError: If the response status code indicates an error.
            requests.RequestException: If a network-level error occurs.
        """
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
        """Display an interactive map showing the station's location.

        Renders a Plotly scatter-map centred on the station. The hover tooltip shows the
        latest river level, current state and trend.

        Raises:
            ValueError: If the station's latitude or longitude is not available.
        """
        if self.lat is None or self.lon is None:
            raise ValueError("Station coordinates are missing.")

        label = self.label or f"Station {self.station_id}"

        fig = px.scatter_map(
            {
                "lat": [self.lat],
                "lon": [self.lon],
                "level": [self.latest_reading.level if self.latest_reading else None],
                "state": [self.state.title()],
                "trend": [self.trend.title()],
            },
            lat="lat",
            lon="lon",
            custom_data=["level", "state", "trend"],
            zoom=14,
            title=label,
            subtitle=self._build_subtitle(),
        )

        fig.update_traces(
            hovertemplate=(
                "Level: %{customdata[0]:.3f}m<br>"
                "State: %{customdata[1]}<br>"
                "Trend: %{customdata[2]}"
            ),
            marker={"size": 12, "color": "dodgerblue"},
        )

        fig.show()

    def plot_chart(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        days: int | None = None,
    ) -> None:
        """Display an interactive time-series chart of river-level readings.

        Fetches readings for the requested period and renders a filled Plotly line
        chart. Where available, reference lines for the all-time maximum and minimum
        are drawn, and the typical level range is highlighted as a shaded band.

        The `start`, `end` and `days` arguments are forwarded directly to
        `get_readings`; their validation rules apply here too.

        Args:
            start: Inclusive start of the time range.
            end: Inclusive end of the time range.
            days: Number of days of history to chart, counting back from now.

        Raises:
            ValueError: If no readings are available for the requested range, or if
                the `start`/`end`/`days` arguments are invalid.
            requests.HTTPError: If the response status code indicates an error.
            requests.RequestException: If a network-level error occurs.
        """
        readings = self.get_readings(start=start, end=end, days=days)
        if not readings:
            raise ValueError("No readings available for the requested time range.")

        timestamps = [r.timestamp for r in readings]
        levels = [r.level for r in readings]

        label = self.label or f"Station {self.station_id}"

        fig = px.line(
            x=timestamps,
            y=levels,
            labels={"x": "Time", "y": "Level (m)"},
            title=f"River level at {label}",
            subtitle=self._build_subtitle(),
        )

        fig.update_traces(
            hovertemplate="Level: %{y:.3f}m<br>Time: %{x|%H:%M, %d %b %Y}",
            line={"width": 3, "color": "dodgerblue"},
            fill="tozeroy",
        )
        fig.update_layout(hoverdistance=-1)
        fig.update_xaxes(
            tickformat="%H:%M\n%d %b %Y",
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikedash="solid",
            spikecolor="black",
        )

        if self.max_on_record is not None:
            fig.add_hline(
                y=self.max_on_record.level,
                annotation_text=(
                    f"Max on record ({self.max_on_record.level:.2f}m"
                    f" on {self.max_on_record.timestamp:%d %b %Y})"
                ),
                annotation_font_color="black",
                line_width=1,
                line_dash="dot",
                line_color="black",
            )

        if self.min_on_record is not None:
            fig.add_hline(
                y=self.min_on_record.level,
                annotation_text=(
                    f"Min on record ({self.min_on_record.level:.2f}m"
                    f" on {self.min_on_record.timestamp:%d %b %Y})"
                ),
                annotation_font_color="black",
                line_width=1,
                line_dash="dot",
                line_color="black",
            )

        if (
            self.typical_range_low is not None
            and self.typical_range_high is not None
            and self.typical_range_low <= self.typical_range_high
        ):
            fig.add_hrect(
                y0=self.typical_range_low,
                y1=self.typical_range_high,
                annotation_text=(
                    f"Typical range ({self.typical_range_low:.2f}m"
                    f" to {self.typical_range_high:.2f}m)"
                ),
                line_width=0,
                fillcolor="dodgerblue",
                opacity=0.15,
            )

        fig.show()

    def _build_subtitle(self) -> str | None:
        if self.river_name is not None and self.catchment_name is not None:
            return f"{self.river_name}, {self.catchment_name} catchment"
        return None
