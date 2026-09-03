"""Known-reference carrier frequency-offset (CFO) estimation.

This module provides a single, narrowly-scoped estimator:
:func:`estimate_frequency_offset`, which reports how far the dominant
spectral component of a block of samples sits from a **caller-supplied,
known** reference frequency.

The estimator is a thin interpretation layer over the Phase 2A spectral
peak estimator (:func:`iqwav.estimation.estimate_peak_frequency`): the
observed frequency is located by that estimator, which itself reuses
:func:`iqwav.dsp.magnitude_spectrum`. This module contributes only the
reference-relative arithmetic and the reference-specific validation; no
FFT, peak-picking, or refinement logic is duplicated here.

Scope
-----
This is a **narrow, known-reference frequency-offset measurement**. The
sample rate and the reference frequency are both assumed known, and the
result is a single scalar difference::

    frequency_offset_hz = observed_frequency_hz - reference_frequency_hz

It is explicitly **not**:

- carrier recovery, a PLL, or a Costas loop (nothing is tracked across
  time; there is no loop, no feedback, and no retained phase state),
- blind carrier detection or automatic reference-frequency discovery
  (the reference is an input, never inferred or searched for),
- CFO correction (the input samples are never modified, mixed,
  de-rotated, resampled, or filtered, and no corrected signal is
  returned),
- symbol-timing recovery, baud-rate estimation, modulation
  classification, or any framing/FEC concern.

Those remain separate milestones, or are permanently out of scope for
this module.
"""

import math
from dataclasses import dataclass

import numpy as np

from .spectral import estimate_peak_frequency

__all__ = ["FrequencyOffsetEstimate", "estimate_frequency_offset"]


@dataclass(frozen=True)
class FrequencyOffsetEstimate:
    """Result of a single known-reference frequency-offset estimate.

    Attributes:
        observed_frequency_hz: Frequency, in Hz, of the dominant spectral
            component actually measured in the analyzed block, exactly as
            reported by :func:`iqwav.estimation.estimate_peak_frequency`
            for the same ``refine`` setting. Signed for complex input,
            non-negative for real input (see the function docstring).
        reference_frequency_hz: The caller-supplied known reference
            frequency in Hz, echoed back for traceability.
        frequency_offset_hz: ``observed_frequency_hz -
            reference_frequency_hz``, in Hz. Positive means the observed
            component sits above the reference, negative means below.
        bin_frequency_hz: Center frequency, in Hz, of the raw FFT bin that
            produced ``observed_frequency_hz``, with no sub-bin
            refinement. Equal to ``observed_frequency_hz`` when
            ``refined`` is False.
        resolution_hz: The raw FFT bin spacing ``sample_rate / N`` in Hz,
            i.e. the frequency quantum of the analysis before any
            refinement. This is the width of one FFT bin, not an error
            bound.
        refined: Whether ``observed_frequency_hz`` (and therefore
            ``frequency_offset_hz``) includes Phase 2A's sub-bin parabolic
            refinement (True) or is the raw bin center (False).
    """

    observed_frequency_hz: float
    reference_frequency_hz: float
    frequency_offset_hz: float
    bin_frequency_hz: float
    resolution_hz: float
    refined: bool


def _validate_reference_frequency(
    reference_frequency_hz: float,
    sample_rate: float,
    *,
    is_complex: bool,
) -> float:
    """Validate the reference frequency and return it as a float.

    The meaningful range is the range of frequencies the analysis can
    actually report, which follows Phase 2A's conventions: a signed
    two-sided axis ``[-sample_rate/2, +sample_rate/2]`` for complex input,
    and the non-negative axis ``[0, sample_rate/2]`` for real input. A
    reference outside that range names a frequency that cannot be observed
    at this sample rate, so no meaningful offset can be formed against it.
    """
    if not math.isfinite(reference_frequency_hz):
        raise ValueError(
            "reference_frequency_hz must be finite, got "
            f"{reference_frequency_hz!r}."
        )
    nyquist = sample_rate / 2.0
    lower = -nyquist if is_complex else 0.0
    if not lower <= reference_frequency_hz <= nyquist:
        kind = "complex" if is_complex else "real"
        raise ValueError(
            f"reference_frequency_hz must lie within [{lower}, {nyquist}] Hz "
            f"for {kind}-valued samples at sample_rate={sample_rate!r}; a "
            "reference outside the frequency range observable at this sample "
            "rate cannot be compared against an observed frequency, got "
            f"{reference_frequency_hz!r}."
        )
    return float(reference_frequency_hz)


def estimate_frequency_offset(
    samples: np.ndarray,
    sample_rate: float,
    reference_frequency_hz: float,
    *,
    refine: bool = True,
) -> FrequencyOffsetEstimate:
    """Estimate the offset of the dominant component from a known reference.

    Definition:
        The observed frequency is the frequency of the dominant spectral
        component of ``samples``, located by the Phase 2A estimator
        :func:`iqwav.estimation.estimate_peak_frequency` (same FFT, same
        peak selection, same optional sub-bin refinement). The reported
        offset is the plain arithmetic difference::

            frequency_offset_hz = observed_frequency_hz
                                  - reference_frequency_hz

        A positive offset means the observed component lies above the
        supplied reference, a negative offset that it lies below. No
        modular reduction, unwrapping, scaling, or additional sign
        convention is applied to that subtraction.

    Known reference, not carrier recovery:
        ``reference_frequency_hz`` is an *input*: the frequency at which
        the component of interest is expected to appear, for example the
        nominal baseband position of a known carrier. This function does
        not search for, guess, or refine the reference, does not track it
        over time, and closes no loop around it. It is not carrier
        recovery, not a PLL or Costas loop, not blind carrier detection,
        and not automatic reference discovery.

    Signal is never modified:
        ``samples`` is read only. No mixing, de-rotation, resampling,
        filtering, or CFO correction is performed and no corrected signal
        is returned -- the measured offset is the entire output.

    Real vs. complex input:
        The conventions are inherited unchanged from Phase 2A. For
        complex-valued ``samples`` (typical baseband IQ) the full
        two-sided spectrum is searched and the observed frequency is
        **signed**, so ``reference_frequency_hz`` is likewise interpreted
        as a signed two-sided frequency and must lie within
        ``[-sample_rate/2, +sample_rate/2]``.

        For real-valued ``samples`` the spectrum is conjugate-symmetric
        and Phase 2A reports a **non-negative** frequency, so the
        reference must also be non-negative and lie within
        ``[0, sample_rate/2]``. Sign is not a meaningful concept for a
        real-valued tone; pass ``abs(f)``.

    Resolution and refinement:
        ``resolution_hz`` is the raw FFT bin spacing ``sample_rate / N``
        for ``N`` input samples: the frequency quantum of the analysis,
        not an error bound. With ``refine=False`` the observed frequency,
        and hence the offset, is quantized to that bin grid, so for an
        isolated tone the offset error is bounded by ``resolution_hz / 2``
        plus the reference's own distance from the grid. With
        ``refine=True`` (the default) Phase 2A's three-point parabolic
        log-magnitude interpolation is applied unchanged, which typically
        places an isolated sinusoid well inside one bin of its true
        frequency. ``bin_frequency_hz`` always reports the unrefined peak
        bin center, so the raw and refined observations remain
        distinguishable in the result.

    Boundary and Nyquist behavior:
        A reference of exactly ``0`` (DC) or exactly ``±sample_rate/2``
        (Nyquist) is accepted. At the Nyquist boundary the two-sided
        frequency axis wraps: Phase 2A wraps refined estimates into
        ``(-sample_rate/2, +sample_rate/2]``, so a complex component
        sitting essentially at the boundary may be reported at either end
        of the axis, in which case the plain difference above appears as a
        large offset of roughly ``±sample_rate`` rather than a small one.
        This function does not attempt to resolve that inherent
        ambiguity. Keep the expected offset well inside the band, or
        interpret near-Nyquist results with the wrap in mind. For real
        input the observed frequency is folded onto ``[0, sample_rate/2]``
        by Phase 2A, so a real Nyquist tone measures at ``+sample_rate/2``
        and gives a near-zero offset against that reference.

    Interpretation limits:
        A single FFT block is analyzed, with no windowing, averaging, or
        leakage correction, and the result reflects whichever component is
        strongest in that block: interference, a spur, or a stronger
        neighbouring signal will be measured instead. The offset is only
        meaningful when the dominant component really is the one whose
        position ``reference_frequency_hz`` describes. Validation order is
        ``sample_rate``, then ``reference_frequency_hz`` (whose valid range
        depends on both the sample rate and whether the input is real or
        complex), then ``samples``, which is validated by Phase 2A.

    Args:
        samples: 1-D real or complex sample array with at least 4 values.
            All values must be finite, and the signal must not be constant
            (a constant or all-zero signal has no oscillating component to
            locate, and raises ``ValueError``).
        sample_rate: Sampling frequency in Hz. Must be positive and finite.
        reference_frequency_hz: Known reference frequency in Hz. Must be
            finite and within ``[-sample_rate/2, +sample_rate/2]`` for
            complex input, or ``[0, sample_rate/2]`` for real input.
        refine: If True (default), use Phase 2A's sub-bin parabolic
            refinement for the observed frequency. If False, use the raw
            FFT bin center.

    Returns:
        A :class:`FrequencyOffsetEstimate` with the observed frequency, the
        echoed reference, their difference, the unrefined bin center, the
        FFT resolution, and whether refinement was applied.

    Raises:
        ValueError: If ``sample_rate`` is not positive and finite, if
            ``reference_frequency_hz`` is not finite or lies outside the
            observable range for this sample rate and input kind, or if
            ``samples`` is not a one-dimensional array of at least 4 finite
            values or is constant (``samples`` validation is delegated to
            :func:`iqwav.estimation.estimate_peak_frequency`).
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(
            f"sample_rate must be positive and finite, got {sample_rate!r}."
        )
    values = np.asarray(samples)
    reference = _validate_reference_frequency(
        reference_frequency_hz,
        float(sample_rate),
        is_complex=bool(np.iscomplexobj(values)),
    )
    # Phase 2A owns sample validation, spectrum computation, peak selection
    # and sub-bin refinement; this estimator only re-expresses its result
    # relative to the known reference.
    observation = estimate_peak_frequency(values, sample_rate, refine=refine)
    observed = float(observation.frequency_hz)
    return FrequencyOffsetEstimate(
        observed_frequency_hz=observed,
        reference_frequency_hz=reference,
        frequency_offset_hz=observed - reference,
        bin_frequency_hz=float(observation.bin_frequency_hz),
        resolution_hz=float(observation.resolution_hz),
        refined=observation.refined,
    )
