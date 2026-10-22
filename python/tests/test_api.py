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

    cov = dist_corr.distance_covariance(v1, v2, True, True)
    assert cov == pytest.approx(0.0, abs=1e-12)


def test_distance_variance_constant_vector() -> None:
    v = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    var = dist_corr.distance_variance(v)
    assert var == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("v1_binary,v2_binary", [(False, False), (True, False), (True, True)])
def test_parallel_matches_sequential(v1_binary: bool, v2_binary: bool) -> None:
    rng = np.random.default_rng(42)
    n = 10_000
    v1 = rng.integers(0, 2, n).astype(np.float64) if v1_binary else rng.normal(size=n)
    v2 = rng.integers(0, 2, n).astype(np.float64) if v2_binary else rng.normal(size=n)

    for func in (dist_corr.distance_correlation, dist_corr.distance_covariance):
        par = func(v1, v2, v1_binary, v2_binary, parallel=True)
        seq = func(v1, v2, v1_binary, v2_binary, parallel=False)
        assert par == pytest.approx(seq, rel=1e-9)

    assert dist_corr.distance_variance(v1, parallel=True) == pytest.approx(
        dist_corr.distance_variance(v1, parallel=False), rel=1e-9
    )


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
