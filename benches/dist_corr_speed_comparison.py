"""Python benchmark comparing dcor vs dist_corr Python package.

Run with:
    poetry run pytest benches/dist_corr_speed_comparison.py --benchmark-min-rounds=10
"""

import numpy as np
import dcor
import dist_corr
import pytest


def samples(sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data matching the Rust benchmark."""
    v1 = np.sin(np.arange(sample_size, dtype=np.float64))
    v2 = np.cos(np.arange(sample_size, dtype=np.float64))
    return v1, v2


@pytest.fixture
def small_samples():
    return samples(2**10)


@pytest.fixture
def little_samples():
    return samples(2**13)


@pytest.fixture
def medium_samples():
    return samples(2**15)


@pytest.fixture
def big_samples():
    return samples(2**20)


IMPLEMENTATIONS = {
    "dcor": dcor.distance_correlation,
    "dist_corr": dist_corr.distance_correlation,
}


@pytest.mark.parametrize("sample_size", [2**10, 2**13, 2**15, 2**20])
def test_distance_correlation_correctness(sample_size: int) -> None:
    v1, v2 = samples(sample_size)
    dcor_result = dcor.distance_correlation(v1, v2)
    rust_result = dist_corr.distance_correlation(v1, v2)
    assert np.isclose(dcor_result, rust_result, atol=1e-10)


@pytest.mark.parametrize("sample_size", [2**10, 2**13, 2**15, 2**20], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("impl_name", ["dcor", "dist_corr"])
def test_dist_corr_speed_comparison(benchmark, impl_name: str, sample_size: int) -> None:
    v1, v2 = samples(sample_size)
    impl = IMPLEMENTATIONS[impl_name]
    result = benchmark(impl, v1, v2)
    print(f"\nimpl: {impl_name} - n: {sample_size} - dist_corr: {result}")


if __name__ == "__main__":
    print("Running quick test without benchmarking...")
    for name, size in [
        ("Small", 2**10),
        ("Little", 2**13),
        ("Medium", 2**15),
        ("Big", 2**20),
    ]:
        v1, v2 = samples(size)
        dcor_result = dcor.distance_correlation(v1, v2)
        rust_result = dist_corr.distance_correlation(v1, v2)
        print(f"{name} (n={size}) -> dcor: {dcor_result}, dist_corr: {rust_result}")
