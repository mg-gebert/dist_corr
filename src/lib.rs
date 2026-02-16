//! dist_corr — distance correlation utilities
//!
//! Distance-correlation routines for 1D data.
//!
//! # dist_corr
//!
//! This crate provides small and fast Rust utilities for computing **distance correlation** and **distance covariance** between pairs of numeric vectors, with optimized implementations for binary (0/1) data.
//!
//! These dependence measures were introduced in the seminal paper:
//!
//! > Székely, G. J., Rizzo, M. L., and Bakirov, N. K. (2007).  
//! > "Measuring and testing dependence by correlation of distances."  
//! > *The Annals of Statistics*, **35**(6), 2769–2794.
//!
//! For more information, see also our companion notes:
//!
//! <a href="https://github.com/mg-gebert/dist_corr/blob/master/dist_corr_notes_gebert_lee.pdf" target="_blank" rel="noopener noreferrer">dist_corr_notes_gebert_lee.pdf</a>
//!
//! # Quickstart
//!
//! Basic usage examples.
//!
//! ### Non-binary data
//!
//! ```rust
//! use dist_corr::{DistCorrelation, DistCovariance};
//!
//! // Distance correlation
//! let v1 = vec![1.0, 2.0, 3.0];
//! let v2 = vec![2.0, 4.0, 6.0];
//!
//! let dist_corr = DistCorrelation;
//! let corr = dist_corr.compute(&v1, &v2).unwrap();
//!
//! // Distance covariance
//! let dist_cov = DistCovariance;
//! let cov = dist_cov.compute(&v1, &v2).unwrap();
//! ```
//!
//! The implemented algorithm follows the one described in:
//!
//! > Chaudhuri, A. and Hu, W. (2019).  
//! > "A fast algorithm for computing distance correlation."  
//! > *Computational Statistics & Data Analysis*, **135**, 15–24.
//!
//! The algorithm is of complexity `O(n log n)` where `n` denotes the common length of the two vectors.
//!
//! ### Binary data
//!
//! If one or both vectors are binary (containing only `0.0` or `1.0`), you can opt into faster, specialized routines by using the `compute_binary` method:
//!
//! ```rust
//! use dist_corr::{DistCorrelation, DistCovariance};
//!
//! let v_bin_1 = vec![0.0, 1.0, 0.0, 1.0];
//! let v_bin_2 = vec![0.0, 0.0, 1.0, 1.0];
//! let v_real = vec![0.5, 2.0, 1.0, -0.3];
//!
//! let dist_corr = DistCorrelation;
//! // v1 binary, v2 non-binary
//! let corr = dist_corr.compute_binary(&v_bin_1, &v_real, true, false).unwrap();
//! // v1 and v2 both binary
//! let corr_both_bin = dist_corr.compute_binary(&v_bin_1, &v_bin_2, true, true).unwrap();
//!
//! let dist_cov = DistCovariance;
//! // v1 non-binary, v2 binary
//! let cov_semi_bin = dist_cov.compute_binary(&v_real, &v_bin_1, false, true).unwrap();
//! // v1 and v2 both binary
//! let cov_both_bin = dist_cov.compute_binary(&v_bin_1, &v_bin_2, true, true).unwrap();
//! ```
//! The complexity of the implemented algorithms in the case of binary vectors is
//! 1. `O(n log n)` if one vector is binary but considerably faster than the non-binary implementation above.
//! 2. `O(n)` if both vectors are binary. Actually, in this case the distance correlation reduces to the absolute value of the Pearson correlation or the <a href="https://en.wikipedia.org/wiki/Phi_coefficient" target="_blank" rel="noopener noreferrer"> Matthews correlation coefficient (MCC)</a>.
//!
//!
//! The formulas behind the faster binary implementations are explained here:
//!
//! <a href="https://github.com/mg-gebert/dist_corr/blob/master/dist_corr_notes_gebert_lee.pdf" target="_blank" rel="noopener noreferrer">dist_corr_notes_gebert_lee.pdf</a>
//!
//!
//! ### Calculating the Distance Correlation Matrix
//!
//! In the following example, we efficiently compute the cross distance correlation matrix, which contains the distance correlations between all pairs of vectors from two lists.
//!
//! ```rust
//! use dist_corr::DistCovariance;
//!
//! let list_1 = [[0.1, 1.0, 2.0, 1.0], [0.0,-1.0,1.0,2.0]];
//! let list_2 = [[-0.1, 1.0, -2.0, 1.0], [0.0,1.0,1.0,2.0]];
//!
//! let dist_cov = DistCovariance;
//!
//! //compute distance variances for each list
//! //note that (.compute_var(&v) is faster than calling .compute(&v,&v) )
//! let dist_var_1: Vec<f64> = list_1.iter().map(|v_1| dist_cov.compute_var(v_1).unwrap()).collect();
//! let dist_var_2: Vec<f64> = list_2.iter().map(|v_2| dist_cov.compute_var(v_2).unwrap()).collect();
//!
//! // compute distance correlation of all vectors pairs in list_1 vs list_2
//! let dist_corr_mat: Vec<Vec<f64>> = list_1
//!        .iter()
//!        .zip(dist_var_1.iter())
//!        .map(|(v_1, var_1)| {
//!            let sqrt_var_1 = var_1.sqrt();
//!            list_2
//!                .iter()
//!                .zip(dist_var_2.iter())
//!                .map(|(v_2, var_2)| {
//!                     let sqrt_var_2 = var_2.sqrt();
//!                     let covariance = dist_cov.compute(v_1, v_2).unwrap();
//!                     (covariance / (sqrt_var_1 * sqrt_var_2)).sqrt()
//!                })
//!                .collect()
//!        })
//!        .collect();
//! ```

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Modules

pub mod api;
pub(crate) mod dist_corr;
pub(crate) mod dist_corr_binary;
pub(crate) mod dist_corr_multi;
pub(crate) mod dist_corr_naive;
pub(crate) mod frob_inner_product;
pub(crate) mod grand_mean;
pub(crate) mod ordering;
pub(crate) mod tests;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Export

#[doc(inline)]
pub use api::DistCorrelation;
#[doc(inline)]
pub use api::DistCovariance;
