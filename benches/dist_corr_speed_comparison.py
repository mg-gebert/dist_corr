"""
Python benchmark for dcor.distance_correlation to compare with Rust implementation.
Requires: pip install dcor pytest-benchmark numpy
Run with: pytest benches/dist_corr_speed_comparison.py --benchmark-min-rounds=10
"""

import numpy as np
import dcor
import pytest


def samples(sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data matching the Rust benchmark."""
    v1 = np.sin(np.arange(sample_size, dtype=np.float64))
    v2 = np.cos(np.arange(sample_size, dtype=np.float64))
    return v1, v2


@pytest.fixture
def small_samples():
    return samples(1024)


@pytest.fixture
def little_samples():
    return samples(8013)


@pytest.fixture
def medium_samples():
    return samples(2**15)


@pytest.fixture
def big_samples():
    return samples(2**20)


def test_dist_corr_small(benchmark, small_samples):
    v1, v2 = small_samples
    result = benchmark(dcor.distance_correlation, v1, v2)
    print(f"\nn: 1024 - dist_corr: {result}")


def test_dist_corr_little(benchmark, little_samples):
    v1, v2 = little_samples
    result = benchmark(dcor.distance_correlation, v1, v2)
    print(f"\nn: 8013 - dist_corr: {result}")


def test_dist_corr_medium(benchmark, medium_samples):
    v1, v2 = medium_samples
    result = benchmark(dcor.distance_correlation, v1, v2)
    print(f"\nn: {2**15} - dist_corr: {result}")


def test_dist_corr_big(benchmark, big_samples):
    v1, v2 = big_samples
    result = benchmark(dcor.distance_correlation, v1, v2)
    print(f"\nn: {2**20} - dist_corr: {result}")


if __name__ == "__main__":
    # Quick test run without benchmark framework
    print("Running quick test without benchmarking...")
    for name, size in [
        ("Small", 1024),
        ("Little", 8013),
        ("Medium", 2**15),
        ("Big", 2**20),
    ]:
        v1, v2 = samples(size)
        result = dcor.distance_correlation(v1, v2)
        print(f"{name} (n={size}): {result}")
