"""Unit tests for iqwav.estimation.frequency_offset.estimate_frequency_offset.

These cover the narrow, known-reference CFO measurement only: a known
sample rate, a known reference frequency, and the plain difference
``observed - reference``. No carrier recovery, synchronization, blind
carrier search, or CFO correction is involved.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_frequency_offset
from iqwav.estimation import (
    FrequencyOffsetEstimate,
    estimate_frequency_offset,
    estimate_peak_frequency,
)
from iqwav.modulation import generate_iq_tone, generate_real_tone

FS = 1000.0
DURATION = 1.0  # -> N = 1000 samples, resolution = 1.0 Hz, integer Hz on-bin
N = 1000
REFERENCE = 137.0  # on-bin reference used by most cases


def _offset_iq_tone(
    reference_hz: float,
    cfo_hz: float,
    *,
    amplitude: float = 1.0,
    duration: float = DURATION,
) -> np.ndarray:
    """A complex tone sitting exactly ``cfo_hz`` away from ``reference_hz``.

    Built by generating the tone at the reference frequency and applying a
    known frequency offset with the existing impairment utility, so the
    ground-truth CFO is exact by construction.
    """
    _, samples = generate_iq_tone(
        fs=FS, freq=reference_hz, duration=duration, amplitude=amplitude
    )
    return apply_frequency_offset(samples, FS, cfo_hz)


def test_zero_offset_when_signal_sits_on_the_reference():
    """A tone exactly at the reference must report an offset of zero."""
    _, samples = generate_iq_tone(fs=FS, freq=REFERENCE, duration=DURATION)
    result = estimate_frequency_offset(samples, FS, REFERENCE)
    assert isinstance(result, FrequencyOffsetEstimate)
    assert result.observed_frequency_hz == pytest.approx(REFERENCE, abs=0.01)
    assert result.reference_frequency_hz == REFERENCE
    assert result.frequency_offset_hz == pytest.approx(0.0, abs=0.01)
    assert result.resolution_hz == pytest.approx(FS / N)
    assert result.refined


def test_positive_offset_is_reported_positive():
    """Observed above the reference must give a positive offset."""
    samples = _offset_iq_tone(REFERENCE, 25.0)
    result = estimate_frequency_offset(samples, FS, REFERENCE)
    assert result.frequency_offset_hz == pytest.approx(25.0, abs=0.01)
    assert result.frequency_offset_hz > 0.0
    assert result.observed_frequency_hz == pytest.approx(REFERENCE + 25.0, abs=0.01)


def test_negative_offset_is_reported_negative():
    """Observed below the reference must give a negative offset."""
    samples = _offset_iq_tone(REFERENCE, -25.0)
    result = estimate_frequency_offset(samples, FS, REFERENCE)
    assert result.frequency_offset_hz == pytest.approx(-25.0, abs=0.01)
    assert result.frequency_offset_hz < 0.0
    assert result.observed_frequency_hz == pytest.approx(REFERENCE - 25.0, abs=0.01)


@pytest.mark.parametrize("cfo_hz", [0.0, 25.0, -25.0, 12.4, -12.4, 137.0])
@pytest.mark.parametrize("refine", [True, False])
def test_offset_is_exactly_observed_minus_reference(cfo_hz, refine):
    """The defining identity must hold for every result, refined or not."""
    samples = _offset_iq_tone(REFERENCE, cfo_hz)
    result = estimate_frequency_offset(samples, FS, REFERENCE, refine=refine)
    assert result.frequency_offset_hz == pytest.approx(
        result.observed_frequency_hz - result.reference_frequency_hz, abs=1e-12
    )
    assert result.reference_frequency_hz == REFERENCE


def test_off_bin_offset_refinement_improves_on_raw_bin():
    """Sub-bin refinement must land closer to the true CFO than the bin does."""
    true_cfo = 12.4  # deliberately not a whole number of 1 Hz bins
    samples = _offset_iq_tone(REFERENCE, true_cfo)

    refined = estimate_frequency_offset(samples, FS, REFERENCE, refine=True)
    raw = estimate_frequency_offset(samples, FS, REFERENCE, refine=False)

    refined_error = abs(refined.frequency_offset_hz - true_cfo)
    raw_error = abs(raw.frequency_offset_hz - true_cfo)
    assert refined_error < raw_error
    assert refined_error < 0.25 * refined.resolution_hz
    assert raw_error <= raw.resolution_hz / 2.0 + 1e-9


def test_off_bin_reference_and_off_bin_observation():
    """Neither the reference nor the observation has to fall on a bin."""
    reference = 137.35
    true_cfo = 10.2
    samples = _offset_iq_tone(reference, true_cfo)

    refined = estimate_frequency_offset(samples, FS, reference)
    raw = estimate_frequency_offset(samples, FS, reference, refine=False)

    assert refined.reference_frequency_hz == reference
    assert refined.frequency_offset_hz == pytest.approx(true_cfo, abs=0.25)
    assert abs(raw.frequency_offset_hz - true_cfo) <= raw.resolution_hz / 2.0 + 1e-9


def test_refine_false_reports_the_raw_bin_quantized_offset():
    """Without refinement the observation is the raw bin center."""
    samples = _offset_iq_tone(REFERENCE, 12.4)
    raw = estimate_frequency_offset(samples, FS, REFERENCE, refine=False)
    assert not raw.refined
    assert raw.observed_frequency_hz == raw.bin_frequency_hz
    # 137 + 12.4 = 149.4 Hz, whose nearest 1 Hz bin center is 149 Hz.
    assert raw.observed_frequency_hz == pytest.approx(149.0)
    assert raw.frequency_offset_hz == pytest.approx(12.0)


def test_refined_result_still_reports_the_unrefined_bin_center():
    """``bin_frequency_hz`` must stay the raw bin even when refining."""
    samples = _offset_iq_tone(REFERENCE, 12.4)
    refined = estimate_frequency_offset(samples, FS, REFERENCE, refine=True)
    raw = estimate_frequency_offset(samples, FS, REFERENCE, refine=False)
    assert refined.refined
    assert refined.bin_frequency_hz == raw.bin_frequency_hz
    assert refined.observed_frequency_hz != refined.bin_frequency_hz


@pytest.mark.parametrize("refine", [True, False])
def test_observation_matches_the_phase_2a_peak_estimator(refine):
    """The observed frequency must come from Phase 2A, unchanged."""
    samples = _offset_iq_tone(REFERENCE, 74.7)
    peak = estimate_peak_frequency(samples, FS, refine=refine)
    result = estimate_frequency_offset(samples, FS, REFERENCE, refine=refine)
    assert result.observed_frequency_hz == peak.frequency_hz
    assert result.bin_frequency_hz == peak.bin_frequency_hz
    assert result.resolution_hz == peak.resolution_hz
    assert result.refined == peak.refined


def test_negative_complex_frequencies_keep_their_sign():
    """Complex input uses the signed two-sided convention on both sides."""
    _, samples = generate_iq_tone(fs=FS, freq=-200.0, duration=DURATION)
    on_reference = estimate_frequency_offset(samples, FS, -200.0)
    above_reference = estimate_frequency_offset(samples, FS, -190.0)
    assert on_reference.observed_frequency_hz < 0.0
    assert on_reference.frequency_offset_hz == pytest.approx(0.0, abs=0.01)
    assert above_reference.frequency_offset_hz == pytest.approx(-10.0, abs=0.01)


def test_offset_may_cross_zero_frequency_for_complex_input():
    """A negative observation against a positive reference is valid."""
    _, samples = generate_iq_tone(fs=FS, freq=-5.0, duration=DURATION)
    result = estimate_frequency_offset(samples, FS, 5.0)
    assert result.frequency_offset_hz == pytest.approx(-10.0, abs=0.01)


def test_real_input_uses_the_nonnegative_frequency_convention():
    """Real tones report |f|, so a real "negative" tone gives the same offset."""
    _, positive = generate_real_tone(fs=FS, freq=145.3, duration=DURATION)
    _, negative = generate_real_tone(fs=FS, freq=-145.3, duration=DURATION)
    positive_result = estimate_frequency_offset(positive, FS, REFERENCE)
    negative_result = estimate_frequency_offset(negative, FS, REFERENCE)
    assert positive_result.observed_frequency_hz >= 0.0
    assert positive_result.frequency_offset_hz == pytest.approx(8.3, abs=0.25)
    assert negative_result.frequency_offset_hz == positive_result.frequency_offset_hz


def test_real_input_offset_can_be_negative_against_a_higher_reference():
    """Sign of the offset, not of the frequency, is what carries meaning."""
    _, samples = generate_real_tone(fs=FS, freq=90.0, duration=DURATION)
    result = estimate_frequency_offset(samples, FS, REFERENCE)
    assert result.frequency_offset_hz == pytest.approx(-47.0, abs=0.01)


@pytest.mark.parametrize(
    ("snr_db", "cfo_hz", "tolerance_hz"),
    [(10.0, 7.5, 1.0), (6.0, -18.25, 2.0), (20.0, 33.75, 0.5)],
)
def test_noisy_tone_offset_remains_recoverable(snr_db, cfo_hz, tolerance_hz):
    """A dominant tone under AWGN still yields the known offset."""
    rng = np.random.default_rng(2026)
    noisy = add_awgn(_offset_iq_tone(REFERENCE, cfo_hz), snr_db=snr_db, rng=rng)
    result = estimate_frequency_offset(noisy, FS, REFERENCE)
    assert result.frequency_offset_hz == pytest.approx(cfo_hz, abs=tolerance_hz)


def test_noisy_real_tone_offset_remains_recoverable():
    """The same holds for real-valued input with real AWGN."""
    rng = np.random.default_rng(11)
    _, clean = generate_real_tone(fs=FS, freq=145.3, duration=DURATION)
    noisy = add_awgn(clean, snr_db=10.0, rng=rng)
    result = estimate_frequency_offset(noisy, FS, REFERENCE)
    assert result.frequency_offset_hz == pytest.approx(8.3, abs=1.0)


def test_offset_follows_the_dominant_component():
    """The measurement describes the strongest component in the block."""
    strong = _offset_iq_tone(REFERENCE, 30.0, amplitude=1.0)
    weak = _offset_iq_tone(REFERENCE, -200.0, amplitude=0.25)
    result = estimate_frequency_offset(strong + weak, FS, REFERENCE)
    assert result.frequency_offset_hz == pytest.approx(30.0, abs=0.01)


def test_resolution_follows_the_analysis_length():
    """``resolution_hz`` is the FFT bin spacing of the analyzed block."""
    samples = _offset_iq_tone(200.0, 30.0, duration=0.5)  # N = 500 -> 2 Hz bins
    result = estimate_frequency_offset(samples, FS, 200.0)
    assert samples.size == 500
    assert result.resolution_hz == pytest.approx(FS / 500)
    assert result.frequency_offset_hz == pytest.approx(30.0, abs=0.01)


@pytest.mark.parametrize("refine", [True, False])
def test_minimum_length_block_is_accepted(refine):
    """Four samples is the documented minimum, and it must not raise."""
    _, samples = generate_iq_tone(fs=FS, freq=250.0, duration=0.004)
    result = estimate_frequency_offset(samples, FS, 250.0, refine=refine)
    assert samples.size == 4
    assert result.resolution_hz == pytest.approx(FS / 4)
    assert result.frequency_offset_hz == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("refine", [True, False])
def test_input_samples_are_not_modified(refine):
    """The estimator measures only; it never corrects or mutates the signal."""
    complex_samples = _offset_iq_tone(REFERENCE, 19.0)
    complex_before = complex_samples.copy()
    estimate_frequency_offset(complex_samples, FS, REFERENCE, refine=refine)
    assert np.array_equal(complex_samples, complex_before)

    _, real_samples = generate_real_tone(fs=FS, freq=145.0, duration=DURATION)
    real_before = real_samples.copy()
    estimate_frequency_offset(real_samples, FS, REFERENCE, refine=refine)
    assert np.array_equal(real_samples, real_before)


def test_dc_reference_is_accepted():
    """A reference of exactly 0 Hz is a valid boundary value."""
    _, samples = generate_iq_tone(fs=FS, freq=3.0, duration=DURATION)
    result = estimate_frequency_offset(samples, FS, 0.0)
    assert result.reference_frequency_hz == 0.0
    assert result.frequency_offset_hz == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("reference_hz", [FS / 2.0, -FS / 2.0])
def test_complex_nyquist_boundary_reference_is_accepted(reference_hz):
    """Both signed Nyquist endpoints are inside the complex reference range."""
    _, samples = generate_iq_tone(fs=FS, freq=400.0, duration=DURATION)
    result = estimate_frequency_offset(samples, FS, reference_hz)
    assert result.reference_frequency_hz == reference_hz
    assert result.frequency_offset_hz == pytest.approx(400.0 - reference_hz, abs=0.01)


@pytest.mark.parametrize("refine", [True, False])
def test_real_nyquist_tone_measures_at_nyquist(refine):
    """A real tone at fs/2 folds to +fs/2, giving a zero offset there."""
    _, samples = generate_real_tone(fs=FS, freq=FS / 2.0, duration=DURATION)
    result = estimate_frequency_offset(samples, FS, FS / 2.0, refine=refine)
    assert result.observed_frequency_hz == pytest.approx(FS / 2.0)
    assert result.frequency_offset_hz == pytest.approx(0.0, abs=1e-9)


def test_complex_nyquist_bin_wrap_is_reported_as_a_plain_difference():
    """Near fs/2 the two-sided axis wraps; no unwrapping is applied.

    A complex tone at 499.6 Hz peaks in the Nyquist bin, which
    ``numpy.fft.fftfreq`` labels -500 Hz. Refinement resolves it back to
    ~+499.74 Hz, but the unrefined bin center stays at -500 Hz, so the plain
    ``observed - reference`` difference against a +500 Hz reference shows the
    full -fs wrap rather than a small offset. This documents the inherent
    boundary ambiguity instead of hiding it.
    """
    _, samples = generate_iq_tone(fs=FS, freq=499.6, duration=DURATION)
    refined = estimate_frequency_offset(samples, FS, FS / 2.0)
    raw = estimate_frequency_offset(samples, FS, FS / 2.0, refine=False)
    assert refined.frequency_offset_hz == pytest.approx(-0.4, abs=0.25)
    assert raw.bin_frequency_hz == pytest.approx(-FS / 2.0)
    assert raw.frequency_offset_hz == pytest.approx(-FS)


def test_integer_arguments_are_accepted_and_normalized_to_float():
    """Integer sample rates and references must behave like their floats."""
    _, samples = generate_iq_tone(fs=FS, freq=REFERENCE, duration=DURATION)
    result = estimate_frequency_offset(samples, 1000, 137)
    assert isinstance(result.reference_frequency_hz, float)
    assert result.reference_frequency_hz == 137.0
    assert result.resolution_hz == pytest.approx(1.0)
    assert result.frequency_offset_hz == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("sample_rate", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_sample_rate_raises(sample_rate):
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=0.1)
    with pytest.raises(ValueError):
        estimate_frequency_offset(samples, sample_rate, 10.0)


@pytest.mark.parametrize(
    "reference_hz", [float("nan"), float("inf"), float("-inf")]
)
def test_nonfinite_reference_frequency_raises(reference_hz):
    _, samples = generate_iq_tone(fs=FS, freq=REFERENCE, duration=DURATION)
    with pytest.raises(ValueError):
        estimate_frequency_offset(samples, FS, reference_hz)


@pytest.mark.parametrize(
    "reference_hz", [FS / 2.0 + 0.5, -FS / 2.0 - 0.5, 2 * FS, -2 * FS]
)
def test_complex_reference_outside_the_two_sided_range_raises(reference_hz):
    _, samples = generate_iq_tone(fs=FS, freq=REFERENCE, duration=DURATION)
    with pytest.raises(ValueError):
        estimate_frequency_offset(samples, FS, reference_hz)


@pytest.mark.parametrize("reference_hz", [-0.5, -REFERENCE, FS / 2.0 + 0.5, FS])
def test_real_reference_outside_the_nonnegative_range_raises(reference_hz):
    """Real input has no negative-frequency convention, so |f| is required."""
    _, samples = generate_real_tone(fs=FS, freq=REFERENCE, duration=DURATION)
    with pytest.raises(ValueError):
        estimate_frequency_offset(samples, FS, reference_hz)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([]),
        np.ones(1),
        np.ones(2),
        np.ones(3),
        np.ones((4, 4)),
        np.array([1.0, np.nan, 2.0, 3.0]),
        np.array([1.0, np.inf, 2.0, 3.0]),
        np.array([1.0 + 1.0j, np.nan + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j]),
    ],
)
def test_invalid_sample_arrays_raise(samples):
    with pytest.raises(ValueError):
        estimate_frequency_offset(samples, FS, 10.0)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(16),
        np.zeros(16, dtype=np.complex128),
        np.full(16, 5.0),
        np.full(16, 2.0 + 3.0j),
    ],
)
def test_constant_and_zero_signals_raise(samples):
    """No oscillating component means no observable frequency to offset."""
    with pytest.raises(ValueError):
        estimate_frequency_offset(samples, FS, 0.0)


def test_result_is_a_frozen_dataclass_with_documented_fields():
    _, samples = generate_iq_tone(fs=FS, freq=REFERENCE, duration=DURATION)
    result = estimate_frequency_offset(samples, FS, REFERENCE)
    for field in (
        "observed_frequency_hz",
        "reference_frequency_hz",
        "frequency_offset_hz",
        "bin_frequency_hz",
        "resolution_hz",
        "refined",
    ):
        assert hasattr(result, field)
    with pytest.raises(FrozenInstanceError):
        result.frequency_offset_hz = 0.0
    with pytest.raises(FrozenInstanceError):
        result.reference_frequency_hz = 0.0


def test_repeated_calls_are_deterministic():
    samples = _offset_iq_tone(REFERENCE, 41.6)
    first = estimate_frequency_offset(samples, FS, REFERENCE)
    second = estimate_frequency_offset(samples, FS, REFERENCE)
    assert first == second
