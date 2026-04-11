# Python Distance Correlation Speed Comparison

This file summarizes the latest benchmark run comparing three implementations.

- `ours`: this repository's Rust-backed Python bindings
- `baseline_py`: external Python package baseline
- `baseline_cpp`: external C++ extension baseline

## Scope and dependency isolation

- All benchmark-only third-party dependencies live in `benches/pyproject.toml`.
- Main project Python package metadata in `python/pyproject.toml` does not include benchmark baseline dependencies.
- Local package under test was installed in editable mode from `../python` for the benchmark environment.

## Semantic compatibility and normalization

- Inputs for all implementations are the same 1D `float64` NumPy arrays:
  - `v1 = sin(arange(n))`
  - `v2 = cos(arange(n))`
- `baseline_cpp` returns the squared quantity compared to `baseline_py`/`ours` distance correlation.
- Benchmark script normalizes `baseline_cpp` by applying `sqrt(...)` before timing and correctness checks.

## Threading policy

To keep thread settings aligned across different backends, each run sets all of:

- `BENCH_NUM_THREADS`
- `RAYON_NUM_THREADS`
- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `NUMBA_NUM_THREADS`
- `OMP_DYNAMIC=FALSE`

The benchmark module also maps `BENCH_NUM_THREADS` into these thread env vars at import time.

Important behavior notes:

- For the pairwise function benchmarked here, the Python baseline does not provide a parallel pairwise path (its `COMPILE_PARALLEL` mode is for rowwise APIs), so multicore and single-core runs are nearly identical for that baseline.
- The C++ baseline has OpenMP in matrix APIs, but not in its pairwise vector path.
- For larger vectors (`n=2^15`, `n=2^18`), `baseline_cpp` is intentionally not run due to poor practical scaling of its pairwise path; those table cells are left blank.

Source note (upstream repositories):

- Python baseline: `CompileMode.COMPILE_PARALLEL` exists, but is wired to rowwise internals (`_fast_dcov_avl.py` `rowwise_impls_dict`), while pairwise AVL/mergesort mappings do not include a parallel target (`impls_dict`).
- C++ baseline: OpenMP pragmas are present in matrix functions only (`distance_covariance_matrix`, `distance_correlation_matrix`), not in pairwise `distance_correlation(x, y)`.

## How it was run

From `benches/`:

```bash
poetry run pip install -e ../python --no-deps -v --force-reinstall

# multicore run
BENCH_NUM_THREADS=8 RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMBA_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  poetry run pytest dist_corr_speed_comparison.py --benchmark-min-rounds=10 --benchmark-json dist_corr_speed_comparison_multicore.json

# single-core run
BENCH_NUM_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  poetry run pytest dist_corr_speed_comparison.py --benchmark-min-rounds=10 --benchmark-json dist_corr_speed_comparison_singlecore.json
```

## Benchmark machine

- OS: macOS 26.3.1 (build 25D2128)
- CPU: Apple M1 Pro
- Cores: 8 logical (6 performance cores)
- Memory: 32 GB
- Python: 3.13.12
- NumPy: 2.2.6
- pytest-benchmark: 5.2.3

## Results: Multicore (`threads=8`, sequential run)

Mean runtime (microseconds):

| n | ours (us) | baseline_py (us) | baseline_cpp (us) | ours speedup vs baseline_py | ours speedup vs baseline_cpp |
|---:|---:|---:|---:|---:|---:|
| 2^8  | 21.0255 | 130.8422 | 892.0175 | 6.22x | 42.43x |
| 2^10 | 96.8443 | 675.7882 | 26054.2335 | 6.98x | 269.03x |
| 2^12 | 481.3296 | 3697.2850 | 556570.9167 | 7.68x | 1156.32x |
| 2^13 | 820.8551 | 7604.7632 | 2232343.1250 | 9.26x | 2719.53x |
| 2^15 | 2457.3722 | 51508.1041 |  | 20.96x |  |
| 2^18 | 18468.3482 | 1510708.0500 |  | 81.80x |  |

## Results: Single core (`threads=1`, sequential run)

Mean runtime (microseconds):

| n | ours (us) | baseline_py (us) | baseline_cpp (us) | ours speedup vs baseline_py | ours speedup vs baseline_cpp |
|---:|---:|---:|---:|---:|---:|
| 2^8  | 21.1385 | 131.3314 | 887.2831 | 6.21x | 41.97x |
| 2^10 | 95.6596 | 678.3898 | 25964.4265 | 7.09x | 271.43x |
| 2^12 | 511.8610 | 3739.8972 | 553456.6375 | 7.31x | 1081.26x |
| 2^13 | 1123.6268 | 7661.6776 | 2237482.4792 | 6.82x | 1991.30x |
| 2^15 | 5131.6849 | 51446.8521 |  | 10.03x |  |
| 2^18 | 48527.9500 | 1510864.4750 |  | 31.13x |  |

## Notes

- Correctness checks pass on all tested sizes run for each implementation:
  - `ours` ~= `baseline_py` within `atol=1e-10`
  - `baseline_cpp` raw output ~= (`baseline_py`)^2, and normalized output ~= `baseline_py`.
- In this environment, the C++ baseline required GCC OpenMP toolchain in Poetry env.
- Benchmark JSON artifacts are stored as:
  - `benches/dist_corr_speed_comparison_multicore.json`
  - `benches/dist_corr_speed_comparison_singlecore.json`
