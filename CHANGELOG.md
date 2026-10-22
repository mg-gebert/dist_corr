# Changelog

All notable changes to this project are documented in this file.

The Rust crate (`dist_corr` on crates.io) and the Python package (`dist-corr` on PyPI)
share version numbers. Each release lists changes per package.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

### Rust — released 2026-10-21

#### Changed

- **Breaking:** `DistCorrelation` and `DistCovariance` are no longer unit structs. Build them
  with `DistCorrelation::new(parallel)` / `DistCovariance::new(parallel)`. All compute methods
  run in parallel on the current rayon thread pool when `parallel` is `true`, and sequentially
  otherwise.
- **Breaking:** compute methods return `Result<f64, dist_corr::Error>` instead of
  `Result<f64, Box<dyn std::error::Error>>`. `Error` is an enum with the variants
  `VectorLengthMismatch`, `VectorEmpty` and `VectorNotBinary(Vector)`.
- Updated dependencies: `itertools` 0.15.0, `rayon` 1.12.0, `log` 0.4.34.

#### Removed

- **Breaking:** the `par_compute`, `par_compute_binary` and `par_compute_var` methods. Use
  `new(true)` and the regular compute methods instead.

#### Known issues

- Sorting in the internal ordering step always runs in parallel, even with `parallel = false`.

### Python — unreleased

#### Added

- `distance_correlation`, `distance_covariance` and `distance_variance` take a `parallel`
  argument. It defaults to `True`, which matches the behavior of 0.1.x.

#### Changed

- Built on the Rust crate 0.2.0.
- Error messages for invalid input have new wording, for example
  `Vector lengths must be identical`. The exception types (`ValueError`, `TypeError`) are
  unchanged.

#### Known issues

- With `parallel=False`, the internal sort still runs in parallel (see the Rust known issues).

[Unreleased]: https://github.com/mg-gebert/dist_corr/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mg-gebert/dist_corr/compare/v0.1.1...v0.2.0
