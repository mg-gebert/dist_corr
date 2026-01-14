# dist_corr

This crate provides small and fast Rust utilities for computing **distance correlation** and **distance covariance** between pairs of numeric vectors in $\mathbb{R}^n$, with optimized implementations for binary (0/1) data.

These dependence measures were introduced in the seminal paper:

> Székely, G. J., Rizzo, M. L., and Bakirov, N. K. (2007).  
> "Measuring and testing dependence by correlation of distances."  
> *The Annals of Statistics*, **35**(6), 2769–2794

and are defined the following way.

## Definition

- **Distance covariance**: a measure of dependence between two random vectors. It is the square root of the average product of centered distances and is always non-negative. Distance covariance is zero if and only if the vectors are independent. More precisely:

  For two vectors $v = (v_1, \ldots, v_n) \in \mathbb{R}^n$ and $w = (w_1, \ldots, w_n) \in \mathbb{R}^n$, the distance covariance is defined by:

$$
dCov^2(v,w) = \frac{1}{n^2} \sum_{i=1}^n \sum_{j=1}^n A_{ij} B_{ij}
$$

  where for $i,j = 1,\ldots,n$:

$$
A_{ij} = |v_i - v_j| - \frac{1}{n} \sum_{i=1}^n |v_i - v_j| - \frac{1}{n}\sum_{j=1}^n |v_i - v_j| + \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n |v_i - v_j|
$$

  and $B_{ij}$ is defined similarly using $w$.

- **Distance correlation**: a dependence measure between two random vectors that is zero if and only if the vectors are independent. Returns a value in [0, 1]. More precisely:

$$
\text{dCorr}(v,w) = \left(\frac{\text{dCov}(v,w)}{\text{dCov}(v,v)^{1/2}\text{dCov}(w,w)^{1/2}}\right)^{1/2}
$$


## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
dist_corr = "0.1"
```

Or use a local path during development:

```toml
dist_corr = { path = "../dist_corr" }
```

## Quickstart

Basic usage examples.

### Non-binary data

```rust
use dist_corr::{DistCorrelation, DistCovariance};

// Distance correlation
let v1 = vec![1.0, 2.0, 3.0];
let v2 = vec![2.0, 4.0, 6.0];

let dist_corr = DistCorrelation;
let corr = dist_corr.compute(&v1, &v2).unwrap();

// Distance covariance
let dist_cov = DistCovariance;
let cov = dist_cov.compute(&v1, &v2).unwrap();
```

The implemented algorithm follows the one described in:

> Chaudhuri, A. and Hu, W. (2019).  
> "A fast algorithm for computing distance correlation."  
> *Computational Statistics & Data Analysis*, **135**, 15–24.

The algorithm is of complexity `O(n log n)` where `n` denotes the common length of the two vectors.

### Binary data

If one or both vectors are binary (containing only `0.0` or `1.0`), you can opt into faster, specialized routines by using the `compute_binary` method:

```rust
use dist_corr::{DistCorrelation, DistCovariance};

let v_bin_1 = vec![0.0, 1.0, 0.0, 1.0];
let v_bin_2 = vec![0.0, 0.0, 1.0, 1.0];
let v_real = vec![0.5, 2.0, 1.0, -0.3];

let dist_corr = DistCorrelation;
// v1 binary, v2 non-binary
let corr = dist_corr.compute_binary(&v_bin_1, &v_real, true, false).unwrap();
// v1 and v2 both binary
let corr_both_bin = dist_corr.compute_binary(&v_bin_1, &v_bin_2, true, true).unwrap();

let dist_cov = DistCovariance;
// v1 non-binary, v2 binary
let cov = dist_cov.compute_binary(&v_real, &v_bin_1, false, true).unwrap();
// v1 and v2 both binary
let cov = dist_cov.compute_binary(&v_bin_1, &v_bin_2, true, true).unwrap();
```
The complexity of the implemented algorithms in the case of binary vectors is 
1. `O(n log n)` if one vector is binary but still considerably faster than the non-binary `O(n log n)` implementation above - see speed benchmarks later.
2. `O(n)` if both vectors are binary.

For further details about the formulas used in the binary implementation, see
<a href="https://github.com/mg-gebert/dist_corr/blob/master/dist_corr_notes_gebert_lee.pdf" target="_blank" rel="noopener noreferrer">dist_corr_notes_gebert_lee.pdf</a>.

Notes:
- When both boolean flags are set to `true`, the corresponding slice is validated to contain only `0.0` or `1.0`. If validation fails, an error is returned.
- For distance correlation, the result is clamped to the range `[0.0, 1.0]` before being returned to avoid tiny negative values due to floating-point error.

### Distance variance

For the special case of computing the distance variance of a single vector, use `DistCovariance::compute_var`:

```rust
use dist_corr::DistCovariance;

let v = vec![1.0, 0.0, 1.0];
let dist_var = DistCovariance;
let var = dist_var.compute_var(&v).unwrap();
```

The implementation of the above is considerably faster than calling `DistCovariance::compute(v, v)` provided that the input `v` is not a binary vector.

## Performance and speed benchmarks

- The crate implements specialized algorithms for binary inputs which are often considerably faster than the general-purpose routines.

- Todo: Here speed benchmarks!

## Error handling

All public compute functions return `Result<f64, Box<dyn std::error::Error>>`. Common error conditions:

- Vectors have different lengths: returns error `"Length of v1 and v2 must be identical"`.
- One or both vectors are empty: returns error `"v1 and v2 must not be empty"` or `"v must not be empty"` for variance.
- A vector is declared binary (flag set) but contains other values: returns error `"v1 must be binary (only 0.0 or 1.0)"` or equivalent for `v2`.

Check the returned `Err` and propagate or handle as needed.

## API Reference (summary)

Type: `DistCorrelation`
- `fn compute(&self, v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>>`
- `fn compute_binary(&self, v1: &[f64], v2: &[f64], v1_binary: bool, v2_binary: bool) -> Result<f64, Box<dyn Error>>`

Type: `DistCovariance`
- `fn compute(&self, v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>>`
- `fn compute_binary(&self, v1: &[f64], v2: &[f64], v1_binary: bool, v2_binary: bool) -> Result<f64, Box<dyn Error>>`
- `fn compute_var(&self, v: &[f64]) -> Result<f64, Box<dyn Error>>`

(See the crate docs or source for more implementation details and exact behaviour.)

## License

See the LICENSE file in this repository for license terms.

## Contact / Maintainers

- Maintainer: Martin Gebert and Miru Lee
- If you found a bug or have a feature request, please open an issue with a small reproducible example.