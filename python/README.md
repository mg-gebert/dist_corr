# dist-corr Python bindings

Python bindings for the Rust `dist_corr` library, providing fast computation of **distance correlation**, **distance covariance**, and **distance variance** between pairs of numeric vectors in R^n, with optimized implementations for binary (0/1) data.

## Mathematical definition

For full mathematical definitions and derivations, see the main project [README](https://github.com/mg-gebert/dist_corr/blob/master/README.md#definition).

## Installation

Install from PyPI:

```bash
pip install dist-corr
```

## Input contract

- Inputs must be `numpy.ndarray`.
- Inputs must be one-dimensional, contiguous, and `float64`.
- No list/tuple/pandas conversion is performed in the bindings.
- `NaN` and `inf` values are passed directly to the Rust core.
 - If the binary flag is set to True, the input must contain only 0.0 and 1.0 values.

## Quickstart


Basic usage examples (see API reference below for details).

### Non-binary data

```python
import numpy as np
import dist_corr

# Distance correlation
v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
v2 = np.array([2.0, 4.0, 6.0], dtype=np.float64)

corr = dist_corr.distance_correlation(v1, v2)
print(f"Distance correlation: {corr}")

# Distance covariance
cov = dist_corr.distance_covariance(v1, v2)
print(f"Distance covariance: {cov}")
```

### Binary data

```python
import numpy as np
import dist_corr

v_bin_1 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
v_bin_2 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
v_real = np.array([0.5, 2.0, 1.0, -0.3], dtype=np.float64)

# v1 binary, v2 non-binary
corr = dist_corr.distance_correlation(v_bin_1, v_real, True, False)
print(f"Distance correlation (binary/non-binary): {corr}")

# v1 and v2 both binary
corr_both_bin = dist_corr.distance_correlation(v_bin_1, v_bin_2, True, True)
print(f"Distance correlation (both binary): {corr_both_bin}")

# v1 non-binary, v2 binary
cov_semi_bin = dist_corr.distance_covariance(v_real, v_bin_1, False, True)
print(f"Distance covariance (non-binary/binary): {cov_semi_bin}")

# v1 and v2 both binary
cov_both_bin = dist_corr.distance_covariance(v_bin_1, v_bin_2, True, True)
print(f"Distance covariance (both binary): {cov_both_bin}")
```

### Distance variance

```python
import numpy as np
import dist_corr

v = np.array([1.0, 0.0, 1.0], dtype=np.float64)
var = dist_corr.distance_variance(v)
print(f"Distance variance: {var}")
```

## Performance and speed benchmarks

See the main project README for detailed [benchmarks](https://github.com/mg-gebert/dist_corr/blob/master/README.md#performance-and-speed-benchmarks) comparing this package to other Python and Rust implementations.

## Error handling

All public compute functions return errors for common conditions:

- Vectors have different lengths
- One or both vectors are empty
- A vector is declared binary (flag set) but contains other values

Check the returned error and propagate or handle as needed.
