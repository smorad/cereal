"""Backwards-compatible re-exports for time-series datasets."""
from __future__ import annotations

from .daily_min_temperatures import DailyMinTemperaturesDataset
from .sunspots import SunspotsDataset

__all__ = ["DailyMinTemperaturesDataset", "SunspotsDataset"]
