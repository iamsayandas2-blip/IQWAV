"""Band-limited IQ signal extraction.

This module provides a single, narrowly-scoped primitive:
:func:`extract_band`, which isolates a caller-specified frequency band
``[lower_hz, upper_hz]`` from a complex IQ signal using existing FIR
filtering machinery (:mod:`iqwav.dsp.filters`) combined with a
known-frequency mix-down/mix-up, reusing
:func:`iqwav.dsp.correct_frequency_offset` for the mixing.

Convention
----------
Frequencies are signed baseband offsets from DC, following the usual IQ
convention: negative frequencies lie below DC, positive above. The
requested band may lie entirely below DC, entirely above DC, or straddle
DC. The band center is mixed down to 0 Hz, low-pass filtered at half the
requested bandwidth, and mixed back up to the original center, so the
returned signal remains centered at the original band location (not at
baseband).

Scope
-----
This is **band-limited preprocessing only**. It is explicitly **not**:

- automatic band detection or blind channelization (the band is supplied
  by the caller, never inferred),
- automatic cutoff selection, modulation recognition, symbol-rate
  estimation, CFO estimation, timing recovery, demodulation, FEC,
  deinterleaving, framing, or payload recovery.

Those remain separate, later milestones (or are permanently out of scope
for this module). The filtered output is not assumed to contain only one
transmitter.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .filters import apply_fir_filter, design_lowpass_fir
from .frequency_correction import correct_frequency_offset

__all__ = ["BandExtractionResult", "extract_band"]

_MIN_SAMPLES = 2


@dataclass(frozen=True)
class BandExtractionResult:
    """Result of a single band-limited IQ extraction.

    Attributes:
        samples: The filtered complex IQ samples, same length as the
            input, centered at the original band location.
        sample_rate: The sample rate in Hz, as supplied.
        lower_hz: The requested lower cutoff, in Hz, as supplied.
        upper_hz: The requested upper cutoff, in Hz, as supplied.
        numtaps: The number of taps used for the internal low-pass FIR
            filter.
        input_length: Number of samples in the input.
        output_length: Number of samples in the output (equal to
            ``input_length``).
    """

    samples: npt.NDArray[np.complex128]
    sample_rate: float
    lower_hz: float
    upper_hz: float
    numtaps: int
    input_length: int
    output_length: int


def _validate_iq_samples(samples: np.ndarray) -> npt.NDArray[np.complexfloating]:
    """Validate 1-D non-empty finite complex IQ samples."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    if not np.iscomplexobj(samples):
        raise ValueError("samples must be complex-valued IQ data.")
    return samples


def extract_band(
    samples: np.ndarray,
    sample_rate: float,
    lower_hz: float,
    upper_hz: float,
    numtaps: int = 101,
) -> BandExtractionResult:
    """Isolate a frequency band from a complex IQ signal.

    Mixes the band center to DC, applies a low-pass FIR filter
    (:func:`iqwav.dsp.design_lowpass_fir` +
    :func:`iqwav.dsp.apply_fir_filter`) with a cutoff at half the
    requested bandwidth, then mixes back up to the original band
    location using :func:`iqwav.dsp.correct_frequency_offset`. Works for
    bands entirely above DC, entirely below DC, or straddling DC.

    Args:
        samples: 1-D complex IQ sample array. Must be non-empty with only
            finite values, and have at least ``_MIN_SAMPLES`` samples.
        sample_rate: Sampling frequency in Hz. Must be positive and
            finite.
        lower_hz: Lower band edge, in Hz. Must be finite. May be
            negative.
        upper_hz: Upper band edge, in Hz. Must be finite and satisfy
            ``lower_hz < upper_hz``. May be negative.
        numtaps: Number of taps for the internal low-pass FIR filter.
            Must be an integer >= 2.

    Returns:
        A :class:`BandExtractionResult` holding the filtered complex128
        samples (same length as the input, a new array; the input is not
        modified) plus extraction metadata.

    Raises:
        ValueError: If any argument violates the constraints above, or if
            the requested band is not valid for the supplied sample rate
            (i.e. does not lie within ``(-fs/2, fs/2)`` with
            ``lower_hz < upper_hz``).
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be positive and finite, got {sample_rate!r}."
        )
    for name, value in (("lower_hz", lower_hz), ("upper_hz", upper_hz)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
    if not lower_hz < upper_hz:
        raise ValueError(
            f"must have lower_hz < upper_hz, got lower_hz={lower_hz!r}, "
            f"upper_hz={upper_hz!r}."
        )
    nyquist = sample_rate / 2.0
    if not (-nyquist < lower_hz and upper_hz < nyquist):
        raise ValueError(
            f"band [{lower_hz!r}, {upper_hz!r}] must lie strictly within "
            f"(-fs/2, fs/2) = ({-nyquist}, {nyquist}) for sample_rate="
            f"{sample_rate!r}."
        )

    samples = _validate_iq_samples(samples)
    if samples.shape[0] < _MIN_SAMPLES:
        raise ValueError(
            f"samples must contain at least {_MIN_SAMPLES} values, got "
            f"{samples.shape[0]}."
        )

    center_hz = (lower_hz + upper_hz) / 2.0
    half_bandwidth_hz = (upper_hz - lower_hz) / 2.0
    # Clamp the low-pass cutoff strictly inside (0, fs/2) to satisfy
    # design_lowpass_fir's validation even for edge-tight bands.
    cutoff_hz = min(half_bandwidth_hz, nyquist * 0.999999)
    if cutoff_hz <= 0:
        raise ValueError(
            f"requested band [{lower_hz!r}, {upper_hz!r}] is too narrow to "
            "filter."
        )

    input_length = samples.shape[0]
    baseband = correct_frequency_offset(samples, sample_rate, center_hz)
    taps = design_lowpass_fir(sample_rate, cutoff_hz, numtaps=numtaps)
    filtered_baseband = apply_fir_filter(baseband, taps)
    result_samples = correct_frequency_offset(
        filtered_baseband, sample_rate, -center_hz
    ).astype(np.complex128, copy=False)

    return BandExtractionResult(
        samples=result_samples,
        sample_rate=sample_rate,
        lower_hz=lower_hz,
        upper_hz=upper_hz,
        numtaps=numtaps,
        input_length=input_length,
        output_length=result_samples.shape[0],
    )