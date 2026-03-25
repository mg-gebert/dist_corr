import numpy as np
import pytest

import dist_corr


def test_distance_correlation_basic() -> None:
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v2 = np.array([2.0, 4.0, 6.0], dtype=np.float64)

    corr = dist_corr.distance_correlation(v1, v2)
    assert corr == pytest.approx(1.0, abs=1e-12)


def test_distance_covariance_independent_binary() -> None:
    v1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    v2 = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

    cov = dist_corr.distance_covariance(v1, v2)
    assert cov == pytest.approx(0.0, abs=1e-12)


def test_distance_variance_constant_vector() -> None:
    v = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    var = dist_corr.distance_variance(v)
    assert var == pytest.approx(0.0, abs=1e-12)


def test_length_mismatch_raises_value_error() -> None:
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v2 = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError):
        dist_corr.distance_correlation(v1, v2)


def test_empty_input_raises_value_error() -> None:
    v = np.array([], dtype=np.float64)

    with pytest.raises(ValueError):
        dist_corr.distance_variance(v)


def test_python_list_rejected() -> None:
    with pytest.raises(TypeError):
        dist_corr.distance_correlation([1.0, 2.0], [1.0, 2.0])  # type: ignore[arg-type]


def test_non_contiguous_rejected() -> None:
    v1 = np.arange(10.0, dtype=np.float64)[::2]
    v2 = np.arange(10.0, dtype=np.float64)[::2]

    with pytest.raises(ValueError):
        dist_corr.distance_correlation(v1, v2)
