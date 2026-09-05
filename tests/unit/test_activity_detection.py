"""Unit tests for iqwav.detection.activity."""

import numpy as np
import pytest

from iqwav.detection import detect_activity


def _burst(rng, n, amplitude):
    return amplitude * (rng.normal(size=n) + 1j * rng.normal(size=n))


def test_all_noise_input_has_no_regions():
    rng = np.random.default_rng(0)
    samples = _burst(rng, 4096, amplitude=0.1)
    result = detect_activity(samples, sample_rate=1000.0, window_size=64)
    assert result.regions == ()


def test_all_active_input_single_region_spans_input():
    # A constant-amplitude tone has identical power in every window, so with
    # a low threshold the whole capture is one active region.
    n = np.arange(4096)
    samples = np.exp(1j * 2 * np.pi * 0.01 * n)
    result = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=-1.0)
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.start_sample == 0
    assert region.end_sample == samples.size


def test_one_active_burst_detected():
    rng = np.random.default_rng(2)
    noise = _burst(rng, 4096, amplitude=0.1)
    samples = noise.copy()
    samples[1000:1500] += _burst(rng, 500, amplitude=5.0)
    result = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0)
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.start_sample <= 1000
    assert region.end_sample >= 1500


def test_multiple_active_bursts_detected_separately():
    rng = np.random.default_rng(3)
    samples = _burst(rng, 8192, amplitude=0.1)
    samples[500:900] += _burst(rng, 400, amplitude=5.0)
    samples[4000:4400] += _burst(rng, 400, amplitude=5.0)
    result = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0)
    assert len(result.regions) == 2


def test_bursts_separated_by_silence_are_distinct_regions():
    rng = np.random.default_rng(4)
    samples = _burst(rng, 8192, amplitude=0.1)
    samples[500:900] += _burst(rng, 400, amplitude=5.0)
    samples[6000:6400] += _burst(rng, 400, amplitude=5.0)
    result = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0, merge_gap_samples=0)
    assert len(result.regions) == 2
    assert result.regions[0].end_sample < result.regions[1].start_sample


def test_merge_gap_joins_nearby_regions():
    rng = np.random.default_rng(5)
    samples = _burst(rng, 4096, amplitude=0.1)
    samples[500:900] += _burst(rng, 400, amplitude=5.0)
    samples[950:1300] += _burst(rng, 350, amplitude=5.0)

    no_merge = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0, merge_gap_samples=0)
    merged = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0, merge_gap_samples=128)

    assert len(no_merge.regions) >= 1
    assert len(merged.regions) == 1
    assert merged.regions[0].start_sample <= 500
    assert merged.regions[0].end_sample >= 1300


def test_merge_gap_respects_actual_sample_gap_not_window_count():
    # Two active windows separated by exactly one full inactive window
    # (64 samples) must NOT merge when merge_gap_samples=1, since the
    # actual gap (64 samples) exceeds the requested gap (1 sample).
    rng = np.random.default_rng(42)
    window_size = 64
    samples = _burst(rng, 4 * window_size, amplitude=0.1)
    samples[0:window_size] += _burst(rng, window_size, amplitude=5.0)
    samples[2 * window_size : 3 * window_size] += _burst(rng, window_size, amplitude=5.0)

    result = detect_activity(
        samples,
        sample_rate=1000.0,
        window_size=window_size,
        threshold_db=6.0,
        merge_gap_samples=1,
    )

    assert len(result.regions) == 2
    assert result.regions[0].end_sample == window_size
    assert result.regions[1].start_sample == 2 * window_size


def test_threshold_affects_detection_sensitivity():
    rng = np.random.default_rng(6)
    samples = _burst(rng, 4096, amplitude=0.1)
    samples[1000:1200] += _burst(rng, 200, amplitude=1.5)

    lenient = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=1.0)
    strict = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=40.0)

    assert len(lenient.regions) >= 1
    assert len(strict.regions) == 0


@pytest.mark.parametrize("window_size", [16, 64, 256])
def test_different_window_sizes_detect_the_burst(window_size):
    rng = np.random.default_rng(7)
    samples = _burst(rng, 4096, amplitude=0.1)
    samples[1000:2000] += _burst(rng, 1000, amplitude=5.0)
    result = detect_activity(samples, sample_rate=1000.0, window_size=window_size, threshold_db=6.0)
    assert len(result.regions) >= 1
    assert result.window_size == window_size


def test_short_input_still_works_with_single_window():
    rng = np.random.default_rng(8)
    samples = _burst(rng, 10, amplitude=1.0)
    result = detect_activity(samples, sample_rate=1000.0, window_size=10)
    assert result.window_powers.size == 1


def test_short_input_uneven_final_window():
    rng = np.random.default_rng(9)
    samples = _burst(rng, 100, amplitude=1.0)
    result = detect_activity(samples, sample_rate=1000.0, window_size=30)
    assert result.window_powers.size == 4  # 30, 30, 30, 10


def test_deterministic_output_for_same_input():
    rng = np.random.default_rng(10)
    samples = _burst(rng, 2048, amplitude=0.1)
    samples[500:900] += _burst(rng, 400, amplitude=5.0)
    first = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0)
    second = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0)
    assert first.regions == second.regions
    assert np.array_equal(first.window_powers, second.window_powers)


def test_input_not_mutated():
    rng = np.random.default_rng(11)
    samples = _burst(rng, 1024, amplitude=1.0)
    original = samples.copy()
    detect_activity(samples, sample_rate=1000.0, window_size=64)
    assert np.array_equal(samples, original)


def test_region_boundaries_are_exact_sample_indices():
    rng = np.random.default_rng(12)
    samples = _burst(rng, 1024, amplitude=0.1)
    samples[256:512] += _burst(rng, 256, amplitude=5.0)
    result = detect_activity(samples, sample_rate=1000.0, window_size=64, threshold_db=6.0)
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.start_sample % 64 == 0
    assert region.end_sample % 64 == 0 or region.end_sample == samples.size
    assert region.duration_s == pytest.approx((region.end_sample - region.start_sample) / 1000.0)
    assert region.start_time_s == pytest.approx(region.start_sample / 1000.0)


def test_invalid_inputs_raise():
    rng = np.random.default_rng(13)
    valid = _burst(rng, 256, amplitude=1.0)

    with pytest.raises(ValueError):
        detect_activity(np.ones((4, 4), dtype=complex), sample_rate=1000.0)
    with pytest.raises(ValueError):
        detect_activity(np.array([], dtype=complex), sample_rate=1000.0)
    with pytest.raises(ValueError):
        bad = valid.copy()
        bad[0] = np.nan
        detect_activity(bad, sample_rate=1000.0)
    with pytest.raises(ValueError):
        detect_activity(np.ones(256), sample_rate=1000.0)  # real, not complex
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=0.0)
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=float("nan"))
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=1000.0, window_size=0)
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=1000.0, window_size=1000)
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=1000.0, threshold_db=float("nan"))
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=1000.0, merge_gap_samples=-1)
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=1000.0, noise_percentile=0.0)
    with pytest.raises(ValueError):
        detect_activity(valid, sample_rate=1000.0, noise_percentile=101.0)