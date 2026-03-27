# Python Distance Correlation Speed Comparison

This file summarizes the benchmark run comparing:

- `dcor.distance_correlation`
- `dist_corr.distance_correlation` (Rust-backed Python bindings)

## How it was run

From `benches/`:

```bash
poetry run pip install -e ../python --no-deps -v --force-reinstall
poetry run pytest dist_corr_speed_comparison.py --benchmark-min-rounds=10
```

The benchmark script uses identical `float64` NumPy inputs for both implementations:

- `v1 = sin(arange(n))`
- `v2 = cos(arange(n))`

## Benchmark machine

- OS: macOS 26.3.1 (build 25D2128)
- CPU: Apple M1 Pro
- Cores: 8 logical (6 performance cores)
- Memory: 32 GB
- Python: 3.13.12
- NumPy: 2.4.3
- dcor: 0.7
- pytest-benchmark: 5.2.3

## Results (mean runtime)

| n | dist_corr (us) | dcor (us) | dist_corr speedup |
|---:|---:|---:|---:|
| 2^10 | 94.8787 | 689.8979 | 7.27x |
| 2^13 | 888.7761 | 7,894.7822 | 8.88x |
| 2^15 | 2,619.8902 | 51,993.1210 | 19.85x |
| 2^20 | 81,005.5383 | 20,148,135.2125 | 248.73x |

## Single-core results (mean runtime)

Run command used:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 poetry run pytest dist_corr_speed_comparison.py --benchmark-min-rounds=10
```

| n | dist_corr (us) | dcor (us) | dist_corr speedup |
|---:|---:|---:|---:|
| 2^10 | 95.0558 | 693.3235 | 7.29x |
| 2^13 | 1,146.9803 | 7,895.1412 | 6.88x |
| 2^15 | 5,243.5654 | 51,804.5833 | 9.88x |
| 2^20 | 233,994.3918 | 20,133,720.9960 | 86.04x |

## Notes

- Benchmark output reported 12 passing tests total (including correctness checks).
- Correctness checks verified that `dcor` and `dist_corr` outputs match within `atol=1e-10` on all benchmark sizes.
- `dist_corr` was rebuilt before the run and confirmed as `release` profile (`[optimized]`).
- Largest observed gap was at `n=2^20`, where `dist_corr` ran in ~81 ms and `dcor` in ~20.15 s.
- In single-core mode, `dist_corr` remains faster at all tested sizes, with the largest gap at `n=2^20` (~234 ms vs ~20.13 s).
