"""Finite-length correlation and peak-detection utilities."""

from .core import (
    autocorrelation,
    cross_correlation,
    find_correlation_peaks,
    normalized_autocorrelation,
    normalized_cross_correlation,
)

__all__ = [
    "autocorrelation",
    "cross_correlation",
    "find_correlation_peaks",
    "normalized_autocorrelation",
    "normalized_cross_correlation",
]