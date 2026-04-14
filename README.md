# flood-monitor

A Python library for fetching and visualising river-level data from the [Environment Agency real time flood-monitoring API](https://environment.data.gov.uk/flood-monitoring/doc/reference). It provides a class that fetches and encapsulates metadata and readings for a given measurement station, including the latest level, historical extremes, typical range, current state and trend. Readings can be retrieved as structured data for a configurable time window, or visualised directly as an interactive map or time-series chart.

## Installation

Install from GitHub using pip:

```bash
pip install "flood-monitor @ git+https://github.com/jezhailwood/flood-monitor.git@v0.1.0"
```

Alternatively, add as a dependency in `pyproject.toml`:

```toml
dependencies = [
    "flood-monitor @ git+https://github.com/jezhailwood/flood-monitor.git@v0.1.0",
]
```

Replace `v0.1.0` with the [latest release tag](https://github.com/jezhailwood/flood-monitor/tags).

## Quickstart

```python
from api_client import APIClient
from flood_monitor import MeasurementStation

client = APIClient("https://environment.data.gov.uk")
station = MeasurementStation.from_api(2642, client)

print(station.label)  # eg "Worcester (Barbourne)"
print(station.river_name)  # eg "River Severn"
print(station.catchment_name)  # eg "Worcestershire Middle Severn"
print(station.lon)  # eg "-2.235272"
print(station.lat)  # eg "52.206967"
print(station.latest_reading)  # eg "Reading(timestamp=..., level=0.587)"
print(station.highest_recent)  # eg "Reading(timestamp=..., level=5.735)"
print(station.max_on_record)  # eg "Reading(timestamp=..., level=5.791)"
print(station.min_on_record)  # eg "Reading(timestamp=..., level=0.487)"
print(station.typical_range_high)  # eg "3.35"
print(station.typical_range_low)  # eg "0.548"
print(station.state)  # eg "normal"
print(station.trend)  # eg "steady"

station.plot_map()  # Interactive map showing the station's location.
station.plot_chart(days=7)  # Interactive time-series chart of readings from the last 7 days.
```

## API reference

Full documentation is available at [jezhailwood.github.io/flood-monitor](https://jezhailwood.github.io/flood-monitor).

## Licence

Released under the [MIT Licence](LICENSE).
