"""Controlled carrier-frequency-offset (CFO) correction.

This module provides a single, narrowly-scoped primitive:
:func:`correct_frequency_offset`, which de-rotates complex IQ samples by a
**caller-supplied, known** frequency offset.

Convention
----------
If the received signal is related to the transmitted signal ``s[n]`` by::

    r[n] = s[n] * exp(+j * 2*pi * frequency_offset_hz * n / sample_rate)

then :func:`correct_frequency_offset` returns::

    s_hat[n] = r[n] * exp(-j * 2*pi * frequency_offset_hz * n / sample_rate)

i.e. it multiplies by the complex conjugate of the phasor that
:func:`iqwav.dsp.apply_frequency_offset` uses to *inject* an offset, so
applying that function followed by this one (with the same
``sample_rate`` and offset) reconstructs ``s[n]`` exactly, up to
floating-point rounding. ``n`` runs over the sample indices
``0, 1, ..., N-1`` in array order.

Scope
-----
This is a **narrow, known-offset correction**. The frequency offset is
assumed known and supplied by the caller, and the operation is a single
elementwise complex multiply. It is explicitly **not**:

- blind CFO estimation or spectral peak search (the offset is an input,
  never inferred or searched for; see
  :func:`iqwav.estimation.estimate_frequency_offset` for measurement),
- carrier recovery, a PLL, or a Costas loop (nothing is tracked across
  time; there is no loop, no feedback, and no retained phase state),
- resampling or filtering (the sample rate and sample count are
  unchanged and no filter is applied),
- amplitude normalization (magnitudes are preserved exactly),
- symbol-timing recovery, symbol-rate estimation, modulation
  classification, or any framing/FEC concern.

Those remain separate milestones, or are permanently out of scope for
this module.
"""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["correct_frequency_offset"]


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


def correct_frequency_offset(
    samples: np.ndarray,
    sample_rate: float,
    frequency_offset_hz: float,
) -> npt.NDArray[np.complex128]:
    """Correct a known carrier frequency offset in complex IQ samples.

    Multiplies the samples by
    ``exp(-j * 2*pi*frequency_offset_hz*n/sample_rate)``, undoing a
    ``+j * 2*pi*frequency_offset_hz*n/sample_rate`` offset applied to the
    original signal. See the module docstring for the exact convention.

    Args:
        samples: 1-D complex IQ sample array. Must be non-empty with only
            finite values.
        sample_rate: Sampling frequency in Hz. Must be positive and finite.
        frequency_offset_hz: Known frequency offset to remove, in Hz. Must
            be finite. May be positive, negative, zero, or larger in
            magnitude than the symbol rate.

    Returns:
        A new array of corrected samples, same length, complex128. The
        input array is not modified.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be positive and finite, got {sample_rate!r}."
        )
    if not math.isfinite(frequency_offset_hz):
        raise ValueError(
            f"frequency_offset_hz must be finite, got {frequency_offset_hz!r}."
        )
    samples = _validate_iq_samples(samples)
    n = np.arange(samples.shape[0], dtype=np.float64)
    corrected = samples * np.exp(
        -1j * 2.0 * np.pi * frequency_offset_hz * n / sample_rate
    )
    return corrected.astype(np.complex128, copy=False)