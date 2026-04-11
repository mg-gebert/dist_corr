# dist_corr

This crate provides small and fast Rust utilities for computing **distance correlation**, **distance covariance** and **distance variance** between pairs of numeric vectors in $\mathbb{R}^n$, with optimized implementations for binary (0/1) data.

These quantities were introduced in the seminal paper

> Székely, G. J., Rizzo, M. L., and Bakirov, N. K. (2007).  
> "Measuring and testing dependence by correlation of distances."  
> *The Annals of Statistics*, **35**(6), 2769–2794

and for completeness we provide the definitions in the next section.

## Definition

- **Distance covariance**: a measure of dependence between two random vectors. It is the square root of the average product of centered distances and is always non-negative. Distance covariance is zero if and only if the vectors are independent. More precisely:

  For two vectors $v = (v_1, \ldots, v_n) \in \mathbb{R}^n$ and $w = (w_1, \ldots, w_n) \in \mathbb{R}^n$, the distance covariance is defined by:

$$
\text{dCov}^2(v,w) = \frac{1}{n^2} \sum_{i=1}^n \sum_{j=1}^n A_{ij} B_{ij}
$$

  where for $i,j = 1,\ldots,n$:

$$
A_{ij} = |v_i - v_j| - \frac{1}{n} \sum_{i=1}^n |v_i - v_j| - \frac{1}{n}\sum_{j=1}^n |v_i - v_j| + \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n |v_i - v_j|
$$

  and $B_{ij}$ is defined similarly using $w$.

- **Distance correlation**: a dependence measure between two random vectors that is zero if and only if the vectors are independent. Returns a value in [0, 1]. More precisely:

$$
\text{dCorr}(v,w) = \frac{\text{dCov}(v,w)}{\text{dCov}(v,v)^{1/2}\text{dCov}(w,w)^{1/2}} \geq 0.
$$

We note that $\text{dCov}^2(v,w) \geq 0$ and dCov is its non-negative square root. 

- **Distance variance**: Distance covariance of a vector with itself. 

$$
\text{dVar}(v) = \text{dCov}(v,v).
$$

## Installation

### Rust crate

Add to your `Cargo.toml`:

```toml
[dependencies]
dist_corr = "0.1"
```

### Python bindings

Python bindings are supported via the `dist_corr` package.

Install from PyPI:

Build and install locally from source:

```bash
cd python
poetry install
poetry run maturin develop --release
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
let cov_semi_bin = dist_cov.compute_binary(&v_real, &v_bin_1, false, true).unwrap();
// v1 and v2 both binary
let cov_both_bin = dist_cov.compute_binary(&v_bin_1, &v_bin_2, true, true).unwrap();
```

The complexity of the implemented algorithms in the case of binary vectors is 
1. $O(n \log n)$ if one vector is binary but considerably faster than the non-binary $O(n \log n)$ implementation above - see speed benchmarks later. We call this the semi-binary case.
2. $O(n)$ if both vectors are binary. Actually, in this case the distance correlation is the same as the absolute value of the Pearson correlation or the <a href="https://en.wikipedia.org/wiki/Phi_coefficient" target="_blank" rel="noopener noreferrer"> Matthews correlation coefficient (MCC)</a>. We compute the latter and call this the full-binary case or algorithm.

For further details about the formulas used in the binary implementation, see
<a href="https://github.com/mg-gebert/dist_corr/blob/master/dist_corr_notes_gebert_lee.pdf" target="_blank" rel="noopener noreferrer">dist_corr_notes_gebert_lee.pdf</a>.

Notes:
- When both boolean flags are set to `true`, the corresponding slice is validated to contain only `0.0` or `1.0`. If validation fails, an error is returned.

### Distance variance

For the special case of computing the distance variance of a single vector, use `DistCovariance::compute_var`:

```rust
use dist_corr::DistCovariance;

let v = vec![1.0, 0.0, 1.0];
let dist_var = DistCovariance;
let var = dist_var.compute_var(&v).unwrap();
```

The implementation of the above is considerably faster than calling `DistCovariance::compute(v, v)` provided that the input `v` is not a binary vector.

 ### Calculating the Distance Correlation Matrix

 In the following example, we efficiently compute the cross distance correlation matrix, which contains the distance correlations between all pairs of vectors from two lists.

 ```rust
 use dist_corr::DistCovariance;

 let list_1 = [[0.1, 1.0, 2.0, 1.0], [0.0,-1.0,1.0,2.0]];
 let list_2 = [[-0.1, 1.0, -2.0, 1.0], [0.0,1.0,1.0,2.0]];

 let dist_cov = DistCovariance;

 //compute distance variances for each list
 //note that (.compute_var(&v) is faster than calling .compute(&v,&v) )
 let dist_var_1: Vec<f64> = list_1.iter().map(|v_1| dist_cov.compute_var(v_1).unwrap()).collect();
 let dist_var_2: Vec<f64> = list_2.iter().map(|v_2| dist_cov.compute_var(v_2).unwrap()).collect();

 // compute distance correlation of all vectors pairs in list_1 vs list_2
 let dist_corr_mat: Vec<Vec<f64>> = list_1
        .iter()
        .zip(dist_var_1.iter())
        .map(|(v_1, var_1)| {
            let sqrt_var_1 = var_1.sqrt();
            list_2
                .iter()
                .zip(dist_var_2.iter())
                .map(|(v_2, var_2)| {
                     let sqrt_var_2 = var_2.sqrt();
                     let covariance = dist_cov.compute(v_1, v_2).unwrap();
                     (covariance / (sqrt_var_1 * sqrt_var_2)).sqrt()
                })
                .collect()
        })
        .collect();
 ```

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

| n | standard (s) | semi-binary (s) |
|:---:|:------------:|:---------------:|
| 2^6  | 3.8514e-6 | 1.2830e-6 |
| 2^8  | 1.93810e-5 | 0.57068e-5 |
| 2^10 | 9.70580e-5 | 2.79640e-5 |
| 2^12 | 4.486000e-4 | 1.313100e-4 |
| 2^14 | 1.652700e-3 | 5.121000e-4 |
| 2^16 | 9.675100e-3 | 2.068600e-3 |
| 2^18 | 4.708400e-2 | 1.064600e-2 |
| 2^20 | 2.961400e-1 | 0.4855400e-1 |
| 2^22 | 1.504900 | 0.2533500 |

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

| n | standard (s) | semi-binary (s) | full-binary (s) |
|:---:|:------------:|:---------------:|:---------------:|
| 2^6  | 3.3123e-6  | 0.8548e-6  | 0.0656e-6 |
| 2^8  | 1.4228e-5  | 0.3175e-5  | 0.0258e-5 |
| 2^10 | 6.6114e-5  | 1.1531e-5  | 0.1144e-5 |
| 2^12 | 2.7786e-4  | 0.6089e-4  | 0.0475e-4 |
| 2^14 | 1.0255e-3  | 0.2861e-3  | 0.0452e-3 |
| 2^16 | 4.7995e-3  | 1.2748e-3  | 0.2663e-3 |
| 2^18 | 2.4360e-2  | 0.8102e-2  | 0.1084e-2 |
| 2^20 | 1.3763e-1  | 0.2827e-1  | 0.0438e-1 |
| 2^22 | 6.2279e-1  | 1.1863e-1  | 0.1749e-1 |

---

### Python benchmark: our bindings vs two anonymized external baselines

For the python benchmark we use the pair of vectors

- $v_1(j) := sin(j)$
- $v_2(j) := cos(j)$

for $j \in 1,...,n$ where $n = 2^{10}, 2^{13}, 2^{15}, 2^{20}$. 

Benchmark machine: macOS 26.3.1 (build 25D2128), Apple M1 Pro, 8 logical cores (6 performance cores), 32 GB RAM, Python 3.13.12, NumPy 2.4.3, dcor 0.7.

Benchmark labels:

- `ours`: this repository's Rust-backed Python bindings.
- `baseline_py`: external Python baseline.
- `baseline_cpp`: external C++ extension baseline.

For comparability, all implementations are called on the same 1D `float64` vectors. The C++ baseline reports the squared quantity, so we apply `sqrt(...)` before comparison.

For larger vectors (`2^15`, `2^18`), the C++ baseline is not run in the pairwise benchmark due to poor practical scaling of that path, and those cells are left blank.

#### Multicore benchmark (`BENCH_NUM_THREADS=8`)

| n | ours (us) | baseline_py (us) | baseline_cpp (us) | speedup (ours vs baseline_py) | speedup (ours vs baseline_cpp) |
|:---:|:----------------------:|:----------------:|:----------------:|:-----------------------------:|:------------------------------:|
| 2^8  | 21.0255 | 130.8422 | 892.0175 | 6.22x | 42.43x |
| 2^10 | 96.8443 | 675.7882 | 26054.2335 | 6.98x | 269.03x |
| 2^12 | 481.3296 | 3697.2850 | 556570.9167 | 7.68x | 1156.32x |
| 2^13 | 820.8551 | 7604.7632 | 2232343.1250 | 9.26x | 2719.53x |
| 2^15 | 2457.3722 | 51508.1041 |  | 20.96x |  |
| 2^18 | 18468.3482 | 1510708.0500 |  | 81.80x |  |

#### Single-core benchmark (`BENCH_NUM_THREADS=1`)

| n | ours (us) | baseline_py (us) | baseline_cpp (us) | speedup (ours vs baseline_py) | speedup (ours vs baseline_cpp) |
|:---:|:----------------------:|:----------------:|:----------------:|:-----------------------------:|:------------------------------:|
| 2^8  | 21.1385 | 131.3314 | 887.2831 | 6.21x | 41.97x |
| 2^10 | 95.6596 | 678.3898 | 25964.4265 | 7.09x | 271.43x |
| 2^12 | 511.8610 | 3739.8972 | 553456.6375 | 7.31x | 1081.26x |
| 2^13 | 1123.6268 | 7661.6776 | 2237482.4792 | 6.82x | 1991.30x |
| 2^15 | 5131.6849 | 51446.8521 |  | 10.03x |  |
| 2^18 | 48527.9500 | 1510864.4750 |  | 31.13x |  |

Threading note: for the pairwise distance correlation function benchmarked here, the Python baseline does not provide a parallel pairwise path (it has a `COMPILE_PARALLEL` mode for rowwise APIs, not this pairwise API), and the C++ baseline uses OpenMP only for matrix APIs, not pairwise vectors. As a result, multicore and single-core timings are nearly identical for both baselines in this benchmark.


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

This project is licensed under the MIT License.

## Contact / Maintainers

- Maintainer: Martin Gebert and Miru Lee
- If you found a bug or have a feature request, please open an issue with a small reproducible example.
