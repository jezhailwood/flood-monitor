"""A Python library for fetching and visualising river-level data.

It provides a class that fetches and encapsulates Environment Agency real-time
flood-monitoring API metadata and readings for a given measurement station, including
the latest level, historical extremes, typical range, current state and trend. Readings
can be retrieved as structured data for a configurable time window, or visualised
directly as an interactive map or time-series chart.

See `flood_monitor.station` for full usage details.
"""

from .station import MeasurementStation, Reading
