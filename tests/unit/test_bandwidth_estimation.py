"""Unit tests for iqwav.estimation.bandwidth.estimate_occupied_bandwidth."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn
from iqwav.estimation import OccupiedBandwidthEstimate, estimate_occupied_bandwidth
from iqwav.modulation import generate_iq_tone, generate_real_tone

FS = 1000.0
DURATION = 1.0  # -> N = 1000 samples, resolution = 1.0 Hz, on-bin freqs integral


def _resolution(n_samples: int, fs: float = FS) -> float:
    return fs / n_samples


# ---------------------------------------------------------------------------
# 1. Narrowband complex tone
# ---------------------------------------------------------------------------


def test_narrowband_complex_tone_is_single_bin_wide():
    """A pure on-bin tone should occupy essentially one FFT bin."""
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=DURATION)
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    assert isinstance(result, OccupiedBandwidthEstimate)
    res = _resolution(samples.size)
    assert result.bandwidth_hz == pytest.approx(res, abs=1e-6)
    assert result.lower_frequency_hz == pytest.approx(137.0 - res / 2.0, abs=1e-6)
    assert result.upper_frequency_hz == pytest.approx(137.0 + res / 2.0, abs=1e-6)
    assert result.percent_power == 99.0


# ---------------------------------------------------------------------------
# 2. Two known complex spectral components
# ---------------------------------------------------------------------------


def test_two_separated_complex_tones_span_both():
    """Two well-separated tones must produce an interval spanning both."""
    _, low = generate_iq_tone(fs=FS, freq=100.0, duration=DURATION)
    _, high = generate_iq_tone(fs=FS, freq=300.0, duration=DURATION)
    result = estimate_occupied_bandwidth(low + high, FS, percent_power=99.0)
    res = _resolution(low.size)
    # The minimal interval covering >=99% power must contain both tones and
    # therefore span from just below 100 Hz to just above 300 Hz.
    assert result.lower_frequency_hz <= 100.0 + res / 2.0
    assert result.upper_frequency_hz >= 300.0 - res / 2.0
    assert result.bandwidth_hz == pytest.approx(
        result.upper_frequency_hz - result.lower_frequency_hz
    )


def test_two_tones_unequal_power_still_span_both_at_high_percent():
    """At a high percent_power, even a much weaker second tone counts."""
    _, strong = generate_iq_tone(fs=FS, freq=50.0, duration=DURATION, amplitude=1.0)
    _, weak = generate_iq_tone(fs=FS, freq=400.0, duration=DURATION, amplitude=0.3)
    result = estimate_occupied_bandwidth(strong + weak, FS, percent_power=99.9)
    assert result.lower_frequency_hz <= 50.5
    assert result.upper_frequency_hz >= 399.5


# ---------------------------------------------------------------------------
# 3. Broadband / noise-like signal
# ---------------------------------------------------------------------------


def test_broadband_noise_like_signal_occupies_wide_band():
    """AWGN alone (no tone) should spread power across most of the band."""
    rng = np.random.default_rng(7)
    n = 512
    noise = rng.normal(size=n) + 1j * rng.normal(size=n)
    result = estimate_occupied_bandwidth(noise, FS, percent_power=90.0)
    # White noise power is spread roughly uniformly, so occupying 90% of
    # the power should require a substantial fraction of the full band,
    # unlike the single-bin-wide narrowband tone case above.
    assert result.bandwidth_hz > 0.5 * FS


def test_tone_plus_noise_is_much_narrower_than_pure_noise():
    """A dominant narrowband tone in modest noise should stay narrow at
    a moderate percent_power, in contrast to pure broadband noise."""
    rng = np.random.default_rng(3)
    _, tone = generate_iq_tone(fs=FS, freq=222.0, duration=DURATION, amplitude=5.0)
    noisy_tone = add_awgn(tone, snr_db=20.0, rng=rng)
    result = estimate_occupied_bandwidth(noisy_tone, FS, percent_power=90.0)
    assert result.bandwidth_hz < 0.1 * FS


# ---------------------------------------------------------------------------
# 4. Monotonicity in percent_power
# ---------------------------------------------------------------------------


def test_increasing_percent_power_never_shrinks_bandwidth():
    """Bandwidth must be non-decreasing as percent_power increases."""
    _, low = generate_iq_tone(fs=FS, freq=100.0, duration=DURATION, amplitude=1.0)
    _, high = generate_iq_tone(fs=FS, freq=300.0, duration=DURATION, amplitude=0.4)
    signal = low + high
    percentages = [5.0, 25.0, 50.0, 75.0, 90.0, 99.0, 99.9, 100.0]
    previous = -1.0
    for pct in percentages:
        result = estimate_occupied_bandwidth(signal, FS, percent_power=pct)
        assert result.bandwidth_hz >= previous - 1e-9
        previous = result.bandwidth_hz


def test_full_percent_power_spans_entire_analyzed_band():
    """percent_power=100 must include every bin with nonzero power.

    Broadband noise has (numerically) nonzero power scattered across
    every FFT bin, so capturing exactly 100% of the total power requires
    the full analyzed band, unlike a clean on-bin tone (see
    ``test_narrowband_complex_tone_is_single_bin_wide``), whose off-peak
    bins carry only negligible floating-point leakage and can already
    satisfy a 100% threshold on their own.
    """
    rng = np.random.default_rng(5)
    n = 256
    noise = rng.normal(size=n) + 1j * rng.normal(size=n)
    result = estimate_occupied_bandwidth(noise, FS, percent_power=100.0)
    assert result.bandwidth_hz == pytest.approx(FS, rel=1e-6)


def test_full_percent_power_on_clean_bin_tone_can_be_narrow():
    """A clean on-bin tone can already satisfy 100% power in ~1 bin,
    since its off-peak bins carry only negligible numerical leakage."""
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=DURATION)
    result = estimate_occupied_bandwidth(samples, FS, percent_power=100.0)
    res = _resolution(samples.size)
    assert result.bandwidth_hz <= 5 * res


# ---------------------------------------------------------------------------
# 5. Real-valued input
# ---------------------------------------------------------------------------


def test_real_tone_bandwidth_matches_complex_tone_bandwidth():
    """Folding the conjugate-symmetric spectrum should give the same
    single-bin-wide result as the complex case, not a doubled one."""
    _, samples = generate_real_tone(fs=FS, freq=137.0, duration=DURATION)
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    res = _resolution(samples.size)
    assert result.bandwidth_hz == pytest.approx(res, abs=1e-6)
    assert result.lower_frequency_hz >= 0.0
    assert result.upper_frequency_hz <= FS / 2.0


def test_real_signal_bounds_stay_within_non_negative_half():
    """Real-signal boundaries must never go below 0 or above fs/2."""
    rng = np.random.default_rng(11)
    n = 256
    noise = rng.normal(size=n)
    result = estimate_occupied_bandwidth(noise, FS, percent_power=99.0)
    assert result.lower_frequency_hz >= 0.0
    assert result.upper_frequency_hz <= FS / 2.0 + 1e-9


# ---------------------------------------------------------------------------
# 6. Positive and negative complex-frequency content
# ---------------------------------------------------------------------------


def test_negative_frequency_complex_tone_reports_negative_interval():
    """A tone at a negative complex frequency must yield a negative band,
    not be folded into the positive half like the real-input case."""
    _, samples = generate_iq_tone(fs=FS, freq=-137.0, duration=DURATION)
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    assert result.upper_frequency_hz <= 0.0
    assert result.lower_frequency_hz < result.upper_frequency_hz


def test_positive_and_negative_complex_tones_do_not_cancel():
    """Two mirror-image complex tones are independent physical content and
    must both be represented (unlike the real-signal folding case)."""
    _, pos = generate_iq_tone(fs=FS, freq=150.0, duration=DURATION)
    _, neg = generate_iq_tone(fs=FS, freq=-150.0, duration=DURATION)
    result = estimate_occupied_bandwidth(pos + neg, FS, percent_power=99.0)
    assert result.lower_frequency_hz < -140.0
    assert result.upper_frequency_hz > 140.0


# ---------------------------------------------------------------------------
# 7. Even and odd sample counts
# ---------------------------------------------------------------------------


def test_even_sample_count_complex():
    n = 512
    res = _resolution(n)
    on_bin_freq = 50 * res  # exactly bin index 50, avoids spectral leakage
    _, samples = generate_iq_tone(fs=FS, freq=on_bin_freq, duration=n / FS)
    assert samples.size == n
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    assert result.bandwidth_hz == pytest.approx(res, abs=1e-6)


def test_odd_sample_count_complex():
    n = 513
    res = _resolution(n)
    on_bin_freq = 50 * res  # exactly bin index 50, avoids spectral leakage
    _, samples = generate_iq_tone(fs=FS, freq=on_bin_freq, duration=n / FS)
    assert samples.size == n
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    assert result.bandwidth_hz == pytest.approx(res, abs=1e-6)


def test_even_sample_count_real():
    n = 256
    _, samples = generate_real_tone(fs=FS, freq=100.0, duration=n / FS)
    assert samples.size == n
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    assert result.lower_frequency_hz >= 0.0
    assert result.upper_frequency_hz <= FS / 2.0


def test_odd_sample_count_real():
    n = 257
    _, samples = generate_real_tone(fs=FS, freq=100.0, duration=n / FS)
    assert samples.size == n
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    assert result.lower_frequency_hz >= 0.0
    assert result.upper_frequency_hz <= FS / 2.0


# ---------------------------------------------------------------------------
# 8. DC / Nyquist edge cases
# ---------------------------------------------------------------------------


def test_dc_heavy_complex_signal_lower_edge_can_reach_near_zero():
    """A tone very close to DC should pull the lower edge near 0 Hz."""
    n = 1000
    _, samples = generate_iq_tone(fs=FS, freq=FS / n, duration=n / FS)  # 1 Hz
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    res = _resolution(n)
    assert result.lower_frequency_hz == pytest.approx(1.0 - res / 2.0, abs=1e-6)


def test_near_nyquist_real_tone_upper_edge_clamped():
    """A real tone near Nyquist must not push the upper edge past fs/2."""
    n = 1000
    freq = FS / 2.0 - 1.0  # just below Nyquist
    _, samples = generate_real_tone(fs=FS, freq=freq, duration=n / FS)
    result = estimate_occupied_bandwidth(samples, FS, percent_power=99.0)
    assert result.upper_frequency_hz <= FS / 2.0 + 1e-9


def test_even_length_nyquist_bin_is_not_double_counted_for_real_input():
    """A real tone placed exactly at Nyquist (even N) must not have its
    single physical bin's power doubled by the conjugate-symmetry fold."""
    n = 200
    fs = 1000.0
    nyquist = fs / 2.0
    # A signal alternating +A, -A, +A, -A, ... is a pure real Nyquist tone.
    samples = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    result = estimate_occupied_bandwidth(samples, fs, percent_power=99.0)
    res = fs / n
    assert result.upper_frequency_hz == pytest.approx(nyquist, abs=1e-6)
    assert result.bandwidth_hz == pytest.approx(res / 2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. Invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_sample_rate_raises(fs):
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=0.1)
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(samples, fs)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(np.array([]), FS)


def test_insufficient_samples_raise():
    for n in (1, 2, 3):
        with pytest.raises(ValueError):
            estimate_occupied_bandwidth(np.ones(n, dtype=np.complex128), FS)


def test_two_dimensional_samples_raise():
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(np.ones((4, 4)), FS)


def test_nonfinite_samples_raise():
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(np.array([1.0, np.nan, 2.0, 3.0]), FS)
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(
            np.array([1.0 + 1.0j, np.inf + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j]), FS
        )


def test_zero_signal_raises():
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(np.zeros(16), FS)
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(np.zeros(16, dtype=np.complex128), FS)


def test_constant_nonzero_signal_raises():
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(np.full(16, 5.0), FS)
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(np.full(16, 2.0 + 3.0j), FS)


@pytest.mark.parametrize("pct", [0.0, -1.0, 100.1, 200.0, float("nan"), float("inf")])
def test_invalid_percent_power_raises(pct):
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=0.1)
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(samples, FS, percent_power=pct)


def test_result_is_frozen_dataclass_with_documented_fields():
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=DURATION)
    result = estimate_occupied_bandwidth(samples, FS)
    assert hasattr(result, "bandwidth_hz")
    assert hasattr(result, "lower_frequency_hz")
    assert hasattr(result, "upper_frequency_hz")
    assert hasattr(result, "percent_power")
    assert result.bandwidth_hz == pytest.approx(
        result.upper_frequency_hz - result.lower_frequency_hz
    )
    with pytest.raises(Exception):
        result.bandwidth_hz = 0.0  # frozen dataclass must reject mutation