import numpy as np
import pytest

from iqwav.correlation import (
    autocorrelation,
    cross_correlation,
    find_correlation_peaks,
    normalized_autocorrelation,
    normalized_cross_correlation,
)


def test_autocorrelation_matches_known_real_sequence():
    lags, values = autocorrelation(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(lags, [-2, -1, 0, 1, 2])
    np.testing.assert_allclose(values, [3.0, 8.0, 14.0, 8.0, 3.0])


def test_autocorrelation_complex_input_has_conjugate_symmetry():
    samples = np.array([1.0 + 2.0j, -3.0 + 1.0j, 2.0 - 4.0j])
    lags, values = autocorrelation(samples)
    zero_index = np.flatnonzero(lags == 0)[0]

    assert values[zero_index] == pytest.approx(np.sum(np.abs(samples) ** 2))
    np.testing.assert_allclose(values, np.conj(values[::-1]))


def test_cross_correlation_delayed_copy_peaks_at_positive_delay():
    reference = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    delayed = np.pad(reference, (3, 0))
    lags, values = cross_correlation(delayed, reference)

    assert lags[np.argmax(np.abs(values))] == 3


def test_cross_correlation_complex_matches_documented_convention():
    first = np.array([1.0 + 1.0j, 2.0 - 1.0j])
    second = np.array([2.0j, 1.0 - 1.0j])
    lags, values = cross_correlation(first, second)

    np.testing.assert_array_equal(lags, [-1, 0, 1])
    expected = np.array([
        first[0] * np.conj(second[1]),
        first[0] * np.conj(second[0]) + first[1] * np.conj(second[1]),
        first[1] * np.conj(second[0]),
    ])
    np.testing.assert_allclose(values, expected)


def test_normalized_autocorrelation_has_unit_zero_lag_for_complex_signal():
    samples = np.array([1.0 + 2.0j, -3.0j, 2.0 - 1.0j])
    lags, values = normalized_autocorrelation(samples)

    assert values[lags == 0][0] == pytest.approx(1.0)
    assert np.all(np.abs(values) <= 1.0 + 1e-12)


def test_normalized_cross_correlation_identical_overlap_is_one_at_delay():
    reference = np.array([1.0, -2.0, 3.0, -4.0])
    delayed = np.pad(reference, (2, 0))
    lags, values = normalized_cross_correlation(delayed, reference)

    assert values[lags == 2][0] == pytest.approx(1.0)
    assert np.all(np.abs(values) <= 1.0 + 1e-12)


@pytest.mark.parametrize(
    "function, arguments",
    [
        (autocorrelation, (np.array([]),)),
        (autocorrelation, (np.array([[1.0]]),)),
        (cross_correlation, (np.array([1.0]), np.array([np.nan]))),
        (cross_correlation, (np.array(["not numeric"]), np.array([1.0]))),
    ],
)
def test_correlation_rejects_invalid_inputs(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize(
    "function, arguments",
    [
        (normalized_autocorrelation, (np.zeros(3),)),
        (normalized_cross_correlation, (np.array([1.0]), np.zeros(2))),
    ],
)
def test_normalized_correlation_rejects_zero_energy_inputs(function, arguments):
    with pytest.raises(ValueError, match="nonzero energy"):
        function(*arguments)


def test_peak_detection_returns_expected_lag_and_original_complex_value():
    lags = np.arange(-4, 5)
    values = np.array([0.0, 0.1, 0.2, 0.5, 0.1, 0.3, 1.0 + 1.0j, 0.2, 0.1])

    indices, peak_lags, peak_values = find_correlation_peaks(
        values, lags, min_height=0.4
    )

    np.testing.assert_array_equal(indices, [3, 6])
    np.testing.assert_array_equal(peak_lags, [-1, 2])
    np.testing.assert_array_equal(peak_values, [0.5, 1.0 + 1.0j])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_distance": 0},
        {"min_height": -1.0},
        {"min_height": "high"},
        {"prominence": np.inf},
        {"use_magnitude": "yes"},
    ],
)
def test_peak_detection_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        find_correlation_peaks(np.array([0.0, 1.0, 0.0]), np.array([-1, 0, 1]), **kwargs)


def test_peak_detection_rejects_invalid_arrays_and_signed_complex_values():
    with pytest.raises(ValueError):
        find_correlation_peaks(np.array([1.0, 2.0]), np.array([0]))
    with pytest.raises(ValueError):
        find_correlation_peaks(
            np.array([0.0 + 1.0j, 1.0j, 0.0 + 1.0j]),
            np.array([-1, 0, 1]),
            use_magnitude=False,
        )
