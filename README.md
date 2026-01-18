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

The algorithm is of complexity $O(n \log n)$ where $n$ denotes the common length of the two vectors.

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
1. $O(n \log n)$ if one vector is binary but still considerably faster than the non-binary $O(n \log n)$ implementation above - see speed benchmarks later. We call this the semi-binary algorithm.
2. $O(n)$ if both vectors are binary. We call this the full-binary algorithm.

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

In this section we evaluate the performance of the standard $O(n\log n)$ vs the semi-binary vs the full-binary algorithm. Benchmarking was performed on a Windows system equipped with an AMD Ryzen 7 PRO 6850U processor and 32 GB of RAM.

### One binary vector

We generate pairs $(v_1,v_2)$ of vectors of length $n$. The entries of $v_1$ are sampled independently and uniformly from the interval $[-10,10]$. The companion vector $v_2$ is then defined by

- $v_2(j) := 1.0$,  if  $v_1(j) < 0.0$
- $v_2(j) := 0.0$,  otherwise 

for each $j=1,\dots,n$, i.e., we compare a general float vector with a binary vector. We compute the distance correlation for input sizes $n=2^{m}$ with $m\in\{6,8,10,12,14,16,18,20,22\}$ using:

1. the standard $O(n\log n)$ algorithm: `dist_corr.compute(&v_1, &v_2)`.
2. the semi-binary $O(n\log n)$ algorithm for one binary vector: `dist_corr.compute_binary(&v_1, &v_2, false, true)`.

The speed test is performed executing the `cargo bench` test `benches\dist_corr_speed_timing.rs`.

**Table 1 — Median running times (seconds) for general float vs binary**

| $n$ | standard (s) | semi-binary (s) |
|:---:|:------------:|:---------------:|
| $2^6$  | $3.8514\times10^{-6}$ | $1.2830\times10^{-6}$ |
| $2^8$  | $1.93810\times10^{-5}$ | $0.57068\times10^{-5}$ |
| $2^{10}$ | $9.70580\times10^{-5}$ | $2.79640\times10^{-5}$ |
| $2^{12}$ | $4.486000\times10^{-4}$ | $1.313100\times10^{-4}$ |
| $2^{14}$ | $1.652700\times10^{-3}$ | $5.121000\times10^{-4}$ |
| $2^{16}$ | $9.675100\times10^{-3}$ | $2.068600\times10^{-3}$ |
| $2^{18}$ | $4.708400\times10^{-2}$ | $1.064600\times10^{-2}$ |
| $2^{20}$ | $2.961400\times10^{-1}$ | $0.4855400\times10^{-1}$ |
| $2^{22}$ | $1.504900\times10^{0}$ | $0.2533500\times10^{0}$ |

---

### Two binary vectors

We generate pairs $(v_1,v_2)$ of length $n$ where $v_1$ is a random binary (0/1) vector with probability $P(0)=P(1)=0.5$. The companion vector $v_2$ is defined by

- $v_2(j) := v_1(j)$, if $2 | v_1(j)$
- $v_2(j) := 0.0$, otherwise

for each $j=1,\dots,n$.
We compute the distance correlation for the same input sizes $n=2^{m}$ with $m\in\{6,8,10,12,14,16,18,20,22\}$ using:

1. the standard $O(n\log n)$ algorithm: `dist_corr.compute(&v_1, &v_2)`.
2. the semi-binary $O(n\log n)$ algorithm for one binary vectorr: `dist_corr.compute_binary(&v_1, &v_2, false, true)`.
3. the full-binary $O(n)$ algorithm for two binary vectors: `dist_corr.compute_binary(&v_1, &v_2, true, true)`.

The speed test is performed executing the `cargo bench` test `benches\dist_corr_speed_timing.rs`.

**Table 2 — Median running times (seconds) for binary vs binary**

| $n$ | standard (s) | semi-binary (s) | full-binary (s) |
|:---:|:------------:|:---------------:|:---------------:|
| $2^{6}$  | $3.3123\times10^{-6}$  | $0.8548\times10^{-6}$  | $0.0656\times10^{-6}$ |
| $2^{8}$  | $1.4228\times10^{-5}$  | $0.3175\times10^{-5}$  | $0.0258\times10^{-5}$ |
| $2^{10}$ | $6.6114\times10^{-5}$  | $1.1531\times10^{-5}$  | $0.1144\times10^{-5}$ |
| $2^{12}$ | $2.7786\times10^{-4}$  | $0.6089\times10^{-4}$  | $0.0475\times10^{-4}$ |
| $2^{14}$ | $1.0255\times10^{-3}$  | $0.2861\times10^{-3}$  | $0.0452\times10^{-3}$ |
| $2^{16}$ | $4.7995\times10^{-3}$  | $1.2748\times10^{-3}$  | $0.2663\times10^{-3}$ |
| $2^{18}$ | $2.4360\times10^{-2}$  | $0.8102\times10^{-2}$  | $0.1084\times10^{-2}$ |
| $2^{20}$ | $1.3763\times10^{-1}$  | $0.2827\times10^{-1}$  | $0.0438\times10^{-1}$ |
| $2^{22}$ | $6.2279\times10^{-1}$  | $1.1863\times10^{-1}$  | $0.1749\times10^{-1}$ |

---


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