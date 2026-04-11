"""Python benchmark comparing dcor, distance-correlation, and dist_corr.

Run with:
    BENCH_NUM_THREADS=8 poetry run pytest benches/dist_corr_speed_comparison.py --benchmark-min-rounds=10
"""

from __future__ import annotations

import os


THREAD_ENV_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def configure_thread_env() -> int | None:
    raw = os.environ.get("BENCH_NUM_THREADS")
    if raw is None:
        return None
    value = str(int(raw))
    for env_key in THREAD_ENV_VARS:
        os.environ[env_key] = value
    os.environ["OMP_DYNAMIC"] = "FALSE"
    return int(value)


BENCH_NUM_THREADS = configure_thread_env()

import dcor
import distance_correlation as distance_correlation_cpp
import dist_corr
import numpy as np
import pytest


COMMON_SAMPLE_SIZES = [2**8, 2**10, 2**12, 2**13, 2**15, 2**18]
CPP_SAMPLE_SIZES = [2**8, 2**10, 2**12, 2**13]


def samples(sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data matching the Rust benchmark."""
    v1 = np.sin(np.arange(sample_size, dtype=np.float64))
    v2 = np.cos(np.arange(sample_size, dtype=np.float64))
    return v1, v2


def distance_correlation_cpp_normalized(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.sqrt(max(distance_correlation_cpp.distance_correlation(v1, v2), 0.0)))


@pytest.fixture(scope="session", autouse=True)
def benchmark_thread_config() -> None:
    configured = {key: os.environ.get(key, "<unset>") for key in THREAD_ENV_VARS}
    print(f"\nBENCH_NUM_THREADS={BENCH_NUM_THREADS}")
    print(f"thread_env={configured}")


IMPLEMENTATIONS = {
    "dcor": dcor.distance_correlation,
    "distance_correlation_cpp": distance_correlation_cpp_normalized,
    "dist_corr": dist_corr.distance_correlation,
}


@pytest.mark.parametrize("sample_size", COMMON_SAMPLE_SIZES)
def test_distance_correlation_correctness(sample_size: int) -> None:
    v1, v2 = samples(sample_size)
    dcor_result = dcor.distance_correlation(v1, v2)
    rust_result = dist_corr.distance_correlation(v1, v2)
    assert np.isclose(dcor_result, rust_result, atol=1e-10)


@pytest.mark.parametrize("sample_size", CPP_SAMPLE_SIZES)
def test_distance_correlation_cpp_correctness(sample_size: int) -> None:
    v1, v2 = samples(sample_size)
    dcor_result = dcor.distance_correlation(v1, v2)
    cpp_raw = distance_correlation_cpp.distance_correlation(v1, v2)
    cpp_result = distance_correlation_cpp_normalized(v1, v2)
    assert np.isclose(cpp_raw, dcor_result * dcor_result, atol=1e-10)
    assert np.isclose(cpp_result, dcor_result, atol=1e-10)


BENCHMARK_CASES = (
    [("dcor", n) for n in COMMON_SAMPLE_SIZES]
    + [("distance_correlation_cpp", n) for n in CPP_SAMPLE_SIZES]
    + [("dist_corr", n) for n in COMMON_SAMPLE_SIZES]
)
BENCHMARK_CASE_IDS = [f"{impl}-n={n}" for impl, n in BENCHMARK_CASES]


@pytest.mark.parametrize(("impl_name", "sample_size"), BENCHMARK_CASES, ids=BENCHMARK_CASE_IDS)
def test_dist_corr_speed_comparison(benchmark, impl_name: str, sample_size: int) -> None:
    v1, v2 = samples(sample_size)
    impl = IMPLEMENTATIONS[impl_name]
    result = benchmark(impl, v1, v2)
    print(f"\nimpl: {impl_name} - n: {sample_size} - dist_corr: {result}")


if __name__ == "__main__":
    print("Running quick test without benchmarking...")
    for name, size in [
        ("Tiny", 2**8),
        ("Small", 2**10),
        ("Medium", 2**12),
        ("Little", 2**13),
        ("Large", 2**15),
        ("Big", 2**18),
    ]:
        v1, v2 = samples(size)
        dcor_result = dcor.distance_correlation(v1, v2)
        cpp_result = (
            distance_correlation_cpp_normalized(v1, v2)
            if size in CPP_SAMPLE_SIZES
            else None
        )
        rust_result = dist_corr.distance_correlation(v1, v2)
        print(
            f"{name} (n={size}) -> dcor: {dcor_result}, "
            f"distance_correlation_cpp: {cpp_result}, dist_corr: {rust_result}"
        )
