// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use std::error::Error;

use crate::dist_corr_binary::dist_cov_binary;
use crate::dist_corr_binary::dist_cov_one_binary;
use crate::dist_corr_binary::{dist_corr_fast_binary, dist_corr_fast_one_binary};
use crate::dist_corr_fast::{dist_corr_fast, dist_cov_fast, dist_var_fast};

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// API Calls

/// Instance for distance correlation computation.
///
/// # Examples
///
/// Basic construction and compute (default: non-binary):
///
/// ```
/// use dist_corr::DistCorrelation;
///
/// let v1 = vec![1.0, 2.0, 3.0];
/// let v2 = vec![2.0, 4.0, 6.0];
/// let dist_corr = DistCorrelation;
/// let result = dist_corr.compute(&v1, &v2);
/// println!("{:?}", result);
/// ```
///
/// --------------------------------------
/// For binary vectors:
///
/// ```
/// use dist_corr::DistCorrelation;
///
/// let v1 = vec![0.0, 1.0, 0.0];
/// let v2 = vec![1.0, 0.0, 1.0];
/// let dist_corr = DistCorrelation;
/// let result = dist_corr.compute_binary(&v1, &v2, true, true);
/// println!("{:?}", result);
/// ```
#[derive(Clone, Debug)]
pub struct DistCorrelation;

/// Instance for distance covariance computation.
///
/// # Examples
///
/// Basic construction and compute (default: non-binary):
///
/// ```
/// use dist_corr::DistCovariance;
///
/// let v1 = vec![1.0, 2.0, 3.0];
/// let v2 = vec![2.0, 4.0, 6.0];
/// let dist_cov = DistCovariance;
/// let result = dist_cov.compute(&v1, &v2);
/// println!("{:?}", result);
/// ```
///
/// --------------------------------------
/// For binary vectors:
///
/// ```
/// use dist_corr::DistCovariance;
///
/// let v1 = vec![0.0, 1.0, 0.0];
/// let v2 = vec![1.0, 0.0, 1.0];
/// let dist_cov = DistCovariance;
/// let result = dist_cov.compute_binary(&v1, &v2, true, true);
/// println!("{:?}", result);
/// ```
#[derive(Clone, Debug)]
pub struct DistCovariance;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementations

impl DistCorrelation {
    /// Computes the distance correlation between two vectors.
    ///
    /// # Arguments
    ///
    /// * `v1` - A slice of `f64` values representing the first data vector.
    /// * `v2` - A slice of `f64` values representing the second data vector.
    ///
    /// # Returns
    ///
    /// Returns a `f64` representing the distance correlation between the two input vectors. The value will
    /// be in the range `[0.0, 1.0]`, where:
    /// - `0.0` indicates no dependence.
    /// - `1.0` indicates perfect linear dependence.
    ///
    /// # Panics
    ///
    /// The function will panic if:
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use dist_corr::DistCorrelation;
    ///
    /// let v1 = vec![2.0, 1.0, -1.1];
    /// let v2 = vec![1.0, 0.3, 1.4];
    ///
    /// let dist_corr = DistCorrelation;
    /// let result = dist_corr.compute(&v1, &v2);
    ///
    /// println!("{:?}", result);
    /// ```
    pub fn compute(&self, v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
        self.compute_binary(v1, v2, false, false)
    }

    /// Computes the distance correlation between two vectors where at least one is binary, i.e. (0-1) valued.
    ///
    /// # Arguments
    ///
    /// * `v1` - A float valued slice representing the first data vector.
    /// * `v2` - A float valued slice with values either 0.0 or 1.0 representing the second data vector.
    /// * `v1_binary` - A flag indicating if v1 should be a binary vector.
    /// * `v2_binary` - A flag indicating if v2 should be a binary vector.
    ///
    /// # Returns
    ///
    /// Returns a `f64` representing the distance correlation between the two binary input vectors. The value will
    /// be in the range `[0.0, 1.0]`, where:
    /// - `0.0` indicates no dependence.
    /// - `1.0` indicates perfect linear dependence.
    ///
    /// # Panics
    ///
    /// The function will panic if:
    /// - `v1` or `v2` is not 0.0 - 1.0 valued as indicated by v1_binary and v2_binary
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    pub fn compute_binary(
        &self,
        v1: &[f64],
        v2: &[f64],
        v1_binary: bool,
        v2_binary: bool,
    ) -> Result<f64, Box<dyn Error>> {
        if v1.len() != v2.len() {
            return Err("Length of v1 must and v2 must be identical".into());
        }

        if v1.is_empty() {
            return Err("v1 and v2 must not be empty".into());
        }

        if v1_binary && !v1.iter().all(|&x| x == 0.0 || x == 1.0) {
            return Err("v1 must be binary (only 0.0 or 1.0)".into());
        }
        if v2_binary && !v2.iter().all(|&x| x == 0.0 || x == 1.0) {
            return Err("v2 must be binary (only 0.0 or 1.0)".into());
        }
        let result = match (v1_binary, v2_binary) {
            (true, true) => dist_corr_fast_binary(v1, v2),
            (true, false) => dist_corr_fast_one_binary(v1, v2),
            (false, true) => dist_corr_fast_one_binary(v2, v1),
            (false, false) => dist_corr_fast(v1, v2),
        };
        result.map(|dist_corr| dist_corr.clamp(0.0, 1.0))
    }
}

impl DistCovariance {
    /// Computes the distance covariance between two vectors.
    ///
    /// # Arguments
    ///
    /// * `v1` - A slice of `f64` values representing the first data vector.
    /// * `v2` - A slice of `f64` values representing the second data vector.
    ///
    /// # Returns
    ///
    /// Returns a `f64` representing the distance covariance between the two input vectors.
    ///
    /// # Panics
    ///
    /// The function will panic if:
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use dist_corr::DistCovariance;
    ///
    /// let v1 = vec![1.0, 2.0, 3.0];
    /// let v2 = vec![2.0, 4.0, 6.0];
    /// let dist_cov = DistCovariance;
    /// let result = dist_cov.compute(&v1, &v2);
    /// println!("{:?}", result);
    /// ```
    pub fn compute(&self, v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
        self.compute_binary(v1, v2, false, false)
    }

    /// Computes the distance covariance between two vectors where at least one is binary, i.e. (0-1) valued.
    ///
    /// # Arguments
    ///
    /// * `v1` - A float valued slice representing the first data vector.
    /// * `v2` - A float valued slice with values either 0.0 or 1.0 representing the second data vector.
    /// * `v1_binary` - A flag indicating if v1 should be a binary vector.
    /// * `v2_binary` - A flag indicating if v2 should be a binary vector.
    ///
    /// # Returns
    ///
    /// Returns a `f64` representing the distance covariance between the two binary input vectors. The value will
    /// be in the range `[0.0, 1.0]`, where:
    /// - `0.0` indicates no dependence.
    /// - `1.0` indicates perfect linear dependence.
    ///
    /// # Panics
    ///
    /// The function will panic if:
    /// - `v1` or `v2` is not 0.0 - 1.0 valued as indicated by v1_binary and v2_binary
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    pub fn compute_binary(
        &self,
        v1: &[f64],
        v2: &[f64],
        v1_binary: bool,
        v2_binary: bool,
    ) -> Result<f64, Box<dyn Error>> {
        if v1.len() != v2.len() {
            return Err("Length of v1 must and v2 must be identical".into());
        }

        if v1.is_empty() {
            return Err("v1 and v2 must not be empty".into());
        }

        if v1_binary && !v1.iter().all(|&x| x == 0.0 || x == 1.0) {
            return Err("v1 must be binary (only 0.0 or 1.0)".into());
        }
        if v2_binary && !v2.iter().all(|&x| x == 0.0 || x == 1.0) {
            return Err("v2 must be binary (only 0.0 or 1.0)".into());
        }

        match (v1_binary, v2_binary) {
            (true, true) => dist_cov_binary(v1, v2),
            (true, false) => dist_cov_one_binary(v1, v2),
            (false, true) => dist_cov_one_binary(v2, v1),
            (false, false) => dist_cov_fast(v1, v2),
        }
    }

    /// Computes the distance variance of a single vector.
    ///
    /// Distance variance is a measure of the variability of a random variable.
    ///
    /// # Arguments
    ///
    /// * `v` - A slice of `f64` values representing the input data vector.
    ///
    /// # Returns
    ///
    /// A `f64` value representing the distance variance of the input vector.
    /// The result is always non-negative:
    /// - `0.0` indicates that all points in the vector are identical.
    /// - Larger values indicate greater spread in the data.
    ///
    /// This mathod is faster than calling `DistCovariance::compute(v,v)`.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - The input vector `v` is empty.
    pub fn compute_var(v: &[f64]) -> Result<f64, Box<dyn Error>> {
        Ok(dist_var_fast(v))
    }
}
