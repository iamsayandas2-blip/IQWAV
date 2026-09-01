"""Reusable finite-length correlation utilities."""

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

__all__ = [
    "autocorrelation",
    "cross_correlation",
    "find_correlation_peaks",
    "normalized_autocorrelation",
    "normalized_cross_correlation",
]


def _validate_signal(samples: np.ndarray, name: str) -> np.ndarray:
    """Return a validated, one-dimensional finite numeric signal."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.issubdtype(samples.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values.")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must contain only finite values.")
    return samples


def _full_lags(first_length: int, second_length: int) -> npt.NDArray[np.int64]:
    """Return lags matching ``np.correlate(first, second, mode='full')``."""
    return np.arange(-(second_length - 1), first_length, dtype=np.int64)


def _correlation_dtype(
    first: np.ndarray, second: np.ndarray
) -> npt.DTypeLike:
    """Use at least double precision and preserve complex correlation."""
    return np.result_type(first.dtype, second.dtype, np.complex128 if (
        np.iscomplexobj(first) or np.iscomplexobj(second)
    ) else np.float64)


def cross_correlation(
    first: np.ndarray, second: np.ndarray
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Compute the full discrete cross-correlation of two finite signals.

    The returned values implement the convention

    ``r_xy[lag] = sum_n x[n + lag] * conj(y[n])``,

    where terms whose indices fall outside either finite input are omitted.
    The returned lag array is ordered from ``-(len(y) - 1)`` through
    ``len(x) - 1`` and has the same order as the correlation values. Under
    this convention, if ``first`` is a delayed copy of ``second`` by ``d``
    samples, their correlation peak occurs at lag ``+d``.

    Args:
        first: 1-D real or complex finite signal ``x``.
        second: 1-D real or complex finite signal ``y``.

    Returns:
        Integer lags and full correlation values. Real inputs return float64
        values; either complex input returns complex128 values.

    Raises:
        ValueError: If either input is empty, non-numeric, non-finite, or not
            one-dimensional.
    """
    first = _validate_signal(first, "first")
    second = _validate_signal(second, "second")
    dtype = _correlation_dtype(first, second)
    first = first.astype(dtype, copy=False)
    second = second.astype(dtype, copy=False)
    return _full_lags(first.size, second.size), np.correlate(first, second, "full")


def autocorrelation(samples: np.ndarray) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Compute the full discrete autocorrelation of a finite signal.

    This is ``cross_correlation(samples, samples)``. Explicitly,
    ``r_xx[lag] = sum_n x[n + lag] * conj(x[n])`` over valid finite-length
    overlap. Lags range from ``-(N - 1)`` to ``N - 1``; lag zero is at index
    ``N - 1`` and equals ``sum(abs(x)**2)``.

    Args:
        samples: 1-D real or complex finite signal.

    Returns:
        Integer lags and full autocorrelation values.

    Raises:
        ValueError: If ``samples`` is empty, non-numeric, non-finite, or not
            one-dimensional.
    """
    return cross_correlation(samples, samples)


def _overlap_energies(
    first: np.ndarray, second: np.ndarray, lags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return energy of the paired sample regions at every correlation lag."""
    first_energy = np.concatenate(([0.0], np.cumsum(np.abs(first) ** 2)))
    second_energy = np.concatenate(([0.0], np.cumsum(np.abs(second) ** 2)))
    first_start = np.maximum(lags, 0)
    second_start = np.maximum(-lags, 0)
    overlap = np.minimum(first.size - first_start, second.size - second_start)
    first_overlap = (
        first_energy[first_start + overlap] - first_energy[first_start]
    )
    second_overlap = (
        second_energy[second_start + overlap] - second_energy[second_start]
    )
    return first_overlap, second_overlap


def normalized_cross_correlation(
    first: np.ndarray, second: np.ndarray
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Compute overlap-energy-normalized full cross-correlation.

    The unnormalized convention is
    ``r_xy[lag] = sum_n x[n + lag] * conj(y[n])``. Each output is normalized
    by the energy in exactly the samples participating at that lag:

    ``rho_xy[lag] = r_xy[lag] / sqrt(E_x(lag) * E_y(lag))``.

    Thus every defined value has magnitude at most one, and equal overlapping
    segments have magnitude one. If either *entire input* has zero energy,
    normalization is undefined and raises ``ValueError``. (With nonzero
    whole-input energy, every valid finite-length overlap has positive energy
    unless it contains only zero-valued samples; such undefined lag values are
    returned as zero.) Lag ordering and delayed-copy interpretation are the
    same as :func:`cross_correlation`.

    Args:
        first: 1-D real or complex finite signal ``x``.
        second: 1-D real or complex finite signal ``y``.

    Returns:
        Integer lags and normalized full correlation values.

    Raises:
        ValueError: If an input is invalid or either input has zero energy.
    """
    first = _validate_signal(first, "first")
    second = _validate_signal(second, "second")
    dtype = _correlation_dtype(first, second)
    first = first.astype(dtype, copy=False)
    second = second.astype(dtype, copy=False)
    if float(np.sum(np.abs(first) ** 2)) == 0.0:
        raise ValueError("first must have nonzero energy for normalization.")
    if float(np.sum(np.abs(second) ** 2)) == 0.0:
        raise ValueError("second must have nonzero energy for normalization.")

    lags, correlation = cross_correlation(first, second)
    first_energy, second_energy = _overlap_energies(first, second, lags)
    denominator = np.sqrt(first_energy * second_energy)
    normalized = np.zeros_like(correlation)
    np.divide(correlation, denominator, out=normalized, where=denominator > 0.0)
    return lags, normalized


def normalized_autocorrelation(
    samples: np.ndarray,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Compute overlap-energy-normalized full autocorrelation.

    This is :func:`normalized_cross_correlation` with the same signal on both
    sides. At lag zero the result is exactly one for every nonzero-energy
    input. At nonzero lags the denominator uses only the two overlapping
    signal regions, so the result is a bounded complex similarity coefficient
    rather than autocorrelation divided by a single global energy.

    Args:
        samples: 1-D real or complex finite signal with nonzero energy.

    Returns:
        Integer lags and normalized full autocorrelation values.

    Raises:
        ValueError: If ``samples`` is invalid or has zero energy.
    """
    return normalized_cross_correlation(samples, samples)


def find_correlation_peaks(
    correlation: np.ndarray,
    lags: np.ndarray,
    *,
    min_height: float | None = None,
    min_distance: int = 1,
    prominence: float | None = None,
    use_magnitude: bool = True,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], np.ndarray]:
    """Find local extrema that are correlation peaks.

    ``correlation`` and ``lags`` must be one-dimensional, equally sized,
    finite arrays. A returned peak index indexes both inputs; peak lags are
    therefore in the caller's explicit lag convention. By default local maxima
    are found in ``abs(correlation)``, appropriate for complex correlation and
    negative real matches. Set ``use_magnitude=False`` to detect only positive
    maxima of real-valued correlation. ``min_height`` and ``prominence`` apply
    to the selected real-valued search sequence, while ``min_distance`` is a
    minimum separation in *array indices*, as defined by SciPy.

    Args:
        correlation: 1-D real or complex finite correlation values.
        lags: 1-D finite integer lag values aligned with ``correlation``.
        min_height: Optional nonnegative finite minimum peak height.
        min_distance: Positive integer minimum separation between peaks.
        prominence: Optional nonnegative finite minimum prominence.
        use_magnitude: Whether to search magnitude instead of signed values.

    Returns:
        Peak indices, corresponding lags, and original (possibly complex)
        correlation values at the peaks.

    Raises:
        ValueError: If arrays or peak-selection arguments are invalid.
    """
    correlation = _validate_signal(correlation, "correlation")
    lags = np.asarray(lags)
    if lags.ndim != 1:
        raise ValueError(f"lags must be one-dimensional, got shape {lags.shape}.")
    if lags.size != correlation.size:
        raise ValueError("lags must have the same length as correlation.")
    if not np.issubdtype(lags.dtype, np.integer):
        raise ValueError("lags must contain integer values.")
    if not isinstance(min_distance, Integral) or isinstance(min_distance, bool):
        raise ValueError("min_distance must be a positive integer.")
    if min_distance < 1:
        raise ValueError("min_distance must be a positive integer.")
    for name, value in (("min_height", min_height), ("prominence", prominence)):
        if value is not None and (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be a nonnegative finite value or None.")
    if not isinstance(use_magnitude, bool):
        raise ValueError("use_magnitude must be a bool.")
    if not use_magnitude and np.iscomplexobj(correlation):
        raise ValueError("signed peak detection requires real-valued correlation.")

    search_values = np.abs(correlation) if use_magnitude else correlation
    indices, _ = find_peaks(
        search_values,
        height=min_height,
        distance=int(min_distance),
        prominence=prominence,
    )
    return indices.astype(np.int64), lags[indices].astype(np.int64), correlation[indices]