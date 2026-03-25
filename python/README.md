# dist-corr Python bindings

This package provides Python bindings for the Rust `dist_corr` library.

## Development

```bash
poetry install
poetry run maturin develop
poetry run pytest
```

## Input contract

- Inputs must be `numpy.ndarray`.
- Inputs must be one-dimensional, contiguous, and `float64`.
- No list/tuple/pandas conversion is performed in the bindings.
- `NaN` and `inf` values are passed directly to the Rust core.
