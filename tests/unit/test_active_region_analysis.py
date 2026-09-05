"""Unit tests for iqwav.pipeline.active_region_analysis.analyze_active_regions.

These cover the orchestration only: activity detection over the full
capture, followed by an independent per-region call into the existing
controlled receiver workflow. No new detection, estimation, CFO
correction, timing recovery, or demodulation logic is exercised here
beyond what the underlying primitives already provide and already test.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.modulation import bpsk_waveform
from iqwav.pipeline import (
    ActiveRegionAnalysisResult,
    ControlledReceiverResult,
    RegionAnalysisResult,
    analyze_active_regions,
)
from iqwav.pipeline import active_region_analysis as ara_module

N_SYMBOLS = 300
SAMPLE_RATE = 48000.0


def _bits(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, count)


def _bpsk_burst(samples_per_symbol: int = 8, *, seed: int = 1) -> np.ndarray:
    bits = _bits(N_SYMBOLS, seed)
    return bpsk_waveform(bits, samples_per_symbol)


def _noise(rng: np.random.Generator, n: int, amplitude: float = 0.05) -> np.ndarray:
    return amplitude * (rng.normal(size=n) + 1j * rng.normal(size=n))


# --------------------------------------------------------------------------
# No activity
# --------------------------------------------------------------------------


def test_no_active_regions_reports_no_activity():
    rng = np.random.default_rng(0)
    samples = _noise(rng, 4096)
    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64)
    assert isinstance(result, ActiveRegionAnalysisResult)
    assert result.status == "no_activity"
    assert result.regions == ()
    assert result.activity.regions == ()


# --------------------------------------------------------------------------
# One active region
# --------------------------------------------------------------------------


def test_one_active_region_is_analyzed():
    rng = np.random.default_rng(1)
    burst = _bpsk_burst()
    noise_before = _noise(rng, 5000)
    noise_after = _noise(rng, 5000)
    samples = np.concatenate([noise_before, burst, noise_after])

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    assert result.status == "analyzed"
    assert len(result.regions) == 1
    region = result.regions[0]
    assert isinstance(region, RegionAnalysisResult)
    assert region.status == "analyzed"
    assert region.receiver is not None
    assert region.receiver.status in ("complete", "partial", "failed")


# --------------------------------------------------------------------------
# Multiple active regions
# --------------------------------------------------------------------------


def test_multiple_active_regions_each_analyzed_independently():
    rng = np.random.default_rng(2)
    burst_a = _bpsk_burst(seed=10)
    burst_b = _bpsk_burst(seed=11)
    samples = np.concatenate(
        [
            _noise(rng, 5000),
            burst_a,
            _noise(rng, 8000),
            burst_b,
            _noise(rng, 5000),
        ]
    )

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    assert result.status == "analyzed"
    assert len(result.regions) == 2
    for region in result.regions:
        assert region.status == "analyzed"
        assert region.receiver is not None


# --------------------------------------------------------------------------
# Original sample-index mapping and deterministic ordering
# --------------------------------------------------------------------------


def test_region_indices_match_activity_detection():
    rng = np.random.default_rng(3)
    burst = _bpsk_burst()
    samples = np.concatenate([_noise(rng, 5000), burst, _noise(rng, 5000)])

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    assert len(result.regions) == len(result.activity.regions)
    for region_result, detected_region in zip(result.regions, result.activity.regions):
        assert region_result.start_sample == detected_region.start_sample
        assert region_result.end_sample == detected_region.end_sample
        assert region_result.duration_samples == (
            detected_region.end_sample - detected_region.start_sample
        )


def test_regions_are_ordered_by_ascending_start_sample():
    rng = np.random.default_rng(4)
    burst_a = _bpsk_burst(seed=20)
    burst_b = _bpsk_burst(seed=21)
    samples = np.concatenate(
        [
            _noise(rng, 5000),
            burst_a,
            _noise(rng, 8000),
            burst_b,
            _noise(rng, 5000),
        ]
    )

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    starts = [region.start_sample for region in result.regions]
    assert starts == sorted(starts)


# --------------------------------------------------------------------------
# Region extraction without input mutation
# --------------------------------------------------------------------------


def test_input_is_not_mutated():
    rng = np.random.default_rng(5)
    burst = _bpsk_burst()
    samples = np.concatenate([_noise(rng, 5000), burst, _noise(rng, 5000)])
    original = samples.copy()

    analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    assert np.array_equal(samples, original)


# --------------------------------------------------------------------------
# Activity metadata preservation
# --------------------------------------------------------------------------


def test_activity_detection_result_is_preserved():
    rng = np.random.default_rng(6)
    burst = _bpsk_burst()
    samples = np.concatenate([_noise(rng, 5000), burst, _noise(rng, 5000)])

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    assert result.activity.sample_rate == SAMPLE_RATE
    assert result.activity.window_size == 64
    assert result.activity.regions == result.activity.regions  # sanity: present
    assert len(result.activity.regions) >= 1


# --------------------------------------------------------------------------
# A region whose analysis fails without crashing the whole pipeline
# --------------------------------------------------------------------------


def test_region_pipeline_error_is_isolated(monkeypatch):
    rng = np.random.default_rng(7)
    burst_a = _bpsk_burst(seed=30)
    burst_b = _bpsk_burst(seed=31)
    samples = np.concatenate(
        [
            _noise(rng, 5000),
            burst_a,
            _noise(rng, 8000),
            burst_b,
            _noise(rng, 5000),
        ]
    )

    real_pipeline = ara_module.run_controlled_receiver_pipeline
    call_count = {"n": 0}

    def _flaky_pipeline(region_samples, sample_rate, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("synthetic per-region failure")
        return real_pipeline(region_samples, sample_rate, **kwargs)

    monkeypatch.setattr(ara_module, "run_controlled_receiver_pipeline", _flaky_pipeline)

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    assert result.status == "analyzed"
    assert len(result.regions) == 2
    assert result.regions[0].status == "error"
    assert result.regions[0].receiver is None
    assert result.regions[0].failure_reason is not None
    assert result.regions[1].status == "analyzed"
    assert result.regions[1].receiver is not None


# --------------------------------------------------------------------------
# No fabricated bits or parameters when a region genuinely fails
# --------------------------------------------------------------------------


def test_failed_region_has_no_fabricated_bits():
    # A very short, noise-only "active" region (forced via a low
    # threshold) should not be able to produce demodulated bits; the
    # per-region receiver result must reflect that honestly rather than
    # fabricating a value.
    rng = np.random.default_rng(8)
    samples = _noise(rng, 512, amplitude=1.0)
    result = analyze_active_regions(
        samples, SAMPLE_RATE, window_size=32, threshold_db=-5.0
    )
    assert result.status == "analyzed"
    for region in result.regions:
        if region.receiver is not None and region.receiver.status != "complete":
            assert region.receiver.bits is None


# --------------------------------------------------------------------------
# Invalid inputs / empty input / real-valued input
# --------------------------------------------------------------------------


def test_empty_input_raises():
    with pytest.raises(ValueError):
        analyze_active_regions(np.array([], dtype=np.complex128), SAMPLE_RATE)


def test_real_valued_input_raises():
    with pytest.raises(ValueError):
        analyze_active_regions(np.zeros(1024, dtype=np.float64), SAMPLE_RATE)


def test_non_finite_input_raises():
    samples = np.zeros(1024, dtype=np.complex128)
    samples[10] = np.inf
    with pytest.raises(ValueError):
        analyze_active_regions(samples, SAMPLE_RATE)


def test_invalid_sample_rate_raises():
    rng = np.random.default_rng(9)
    samples = _noise(rng, 1024)
    with pytest.raises(ValueError):
        analyze_active_regions(samples, -1.0)


def test_two_dimensional_input_raises():
    samples = np.zeros((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        analyze_active_regions(samples, SAMPLE_RATE)


# --------------------------------------------------------------------------
# Insufficient region length still yields a structured (non-crashing) result
# --------------------------------------------------------------------------


def test_short_active_region_yields_structured_partial_or_failed_result():
    # A brief, isolated spike surrounded by silence creates one short
    # active region, too short for reliable timing recovery; this must
    # not crash and must not fabricate bits.
    rng = np.random.default_rng(10)
    samples = _noise(rng, 4096, amplitude=0.05)
    samples[2000:2032] += _noise(rng, 32, amplitude=5.0)

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=32, threshold_db=6.0)

    assert result.status == "analyzed"
    assert len(result.regions) >= 1
    for region in result.regions:
        assert region.status in ("analyzed", "error")
        if region.receiver is not None and region.receiver.bits is None:
            assert region.receiver.status in ("partial", "failed")


# --------------------------------------------------------------------------
# Regression: exact activity-detected slices are passed downstream
# --------------------------------------------------------------------------


def _minimal_receiver_result(sample_rate: float) -> ControlledReceiverResult:
    """A minimal, structurally valid ControlledReceiverResult stand-in."""
    return ControlledReceiverResult(
        status="failed",
        failure_reason="stub result for regression test",
        sample_rate=sample_rate,
        peak_frequency=None,
        occupied_bandwidth=None,
        snr=None,
        noise_floor=None,
        symbol_rate=None,
        samples_per_symbol=None,
        samples_per_symbol_known=False,
        timing=None,
        timing_offset=None,
        modulation_estimate=None,
        modulation=None,
        modulation_known=False,
        frequency_offset_hz=None,
        cfo_corrected_samples=None,
        bits=None,
        stage_status={},
        failure_reasons={},
    )


def test_exact_region_slices_are_passed_to_receiver_pipeline(monkeypatch):
    rng = np.random.default_rng(42)
    burst_a = _bpsk_burst(seed=40)
    burst_b = _bpsk_burst(seed=41)
    samples = np.concatenate(
        [
            _noise(rng, 5000),
            burst_a,
            _noise(rng, 8000),
            burst_b,
            _noise(rng, 5000),
        ]
    )

    captured_region_samples = []

    def _fake_pipeline(region_samples, sample_rate, **kwargs):
        captured_region_samples.append(region_samples)
        return _minimal_receiver_result(sample_rate)

    monkeypatch.setattr(ara_module, "run_controlled_receiver_pipeline", _fake_pipeline)

    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    assert len(result.regions) >= 2
    assert len(captured_region_samples) == len(result.regions)

    for captured, region_result in zip(captured_region_samples, result.regions):
        expected = samples[region_result.start_sample : region_result.end_sample]
        assert np.array_equal(captured, expected)


# --------------------------------------------------------------------------
# Frozen dataclasses
# --------------------------------------------------------------------------


def test_result_dataclasses_are_frozen():
    rng = np.random.default_rng(11)
    burst = _bpsk_burst()
    samples = np.concatenate([_noise(rng, 5000), burst, _noise(rng, 5000)])
    result = analyze_active_regions(samples, SAMPLE_RATE, window_size=64, threshold_db=6.0)

    with pytest.raises(FrozenInstanceError):
        result.status = "no_activity"  # type: ignore[misc]

    if result.regions:
        with pytest.raises(FrozenInstanceError):
            result.regions[0].status = "error"  # type: ignore[misc]