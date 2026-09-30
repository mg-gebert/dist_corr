// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Modules
pub mod err
{

  /// Error type for computations.
  #[derive(Debug)]
  pub enum Error
  {
    VectorLengthMismatch,
    VectorNotBinary,
    VectorEmpty
  }

  impl std::fmt::Display for Error
  {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
      match self
      {
        Error::VectorLengthMismatch => write!(f, "Vector lengths must be identical"),
        Error::VectorNotBinary => write!(f, "Vector must be binary (only 0.0 or 1.0)"),
        Error::VectorEmpty => write!(f, "Vector must not be empty"),
      }
    }
  }

  impl std::error::Error for Error {}
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use crate::api::err::Error;
use crate::dist_corr::{dist_corr, dist_cov, dist_var};
use crate::dist_corr_binary::{
    dist_corr_both_binary, dist_corr_one_binary, dist_cov_both_binary, dist_cov_one_binary,
};

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// API Calls

/// Instance for distance correlation computation.
#[derive(Clone, Debug)]
pub struct DistCorrelation;

/// Instance for distance covariance computation.
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
    /// # Errors
    ///
    /// The function will return an error if:
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use dist_corr::DistCorrelation;
    ///
    /// let v1 = vec![1.0, 2.0, 3.0];
    /// let v2 = vec![2.0, 4.0, 6.0];
    ///
    /// let dist_corr = DistCorrelation;
    /// let result = dist_corr.compute(&v1, &v2).unwrap();
    ///
    /// assert_eq!(result, 1.0);
    /// ```
    pub fn compute(&self, v1: &[f64], v2: &[f64]) -> Result<f64, Error> {
        self.compute_binary(v1, v2, false, false)
    }

    /// Computes the distance correlation between two vectors where at least one is binary, i.e. 0-1-valued.
    ///
    /// # Arguments
    ///
    /// * `v1` - A slice of `f64` values representing the first data vector.
    /// * `v2` - A slice of `f64` values representing the second data vector.
    /// * `v1_binary` - A flag indicating if v1 is a binary vector, i.e. with values either 0.0 or 1.0.
    /// * `v2_binary` - A flag indicating if v2 is a binary vector, i.e. with values either 0.0 or 1.0.
    ///
    /// # Returns
    ///
    /// Returns a `f64` representing the distance correlation between the two binary input vectors. The value will
    /// be in the range `[0.0, 1.0]`, where:
    /// - `0.0` indicates no dependence.
    /// - `1.0` indicates perfect linear dependence.
    ///
    /// # Errors
    ///
    /// The function will return an error if:
    /// - `v1` or `v2` is not 0-1-valued as indicated by v1_binary and v2_binary
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    ///
    /// ```
    /// use dist_corr::DistCorrelation;
    ///
    /// let v1 = vec![0.0, 1.0, 0.0, 1.0];
    /// let v2 = vec![0.0, 1.0, 1.0, 0.0];
    ///
    /// let dist_corr = DistCorrelation;
    /// let result = dist_corr.compute_binary(&v1, &v2, true, true).unwrap();
    ///
    /// assert_eq!(result, 0.0);
    /// ```
    pub fn compute_binary(
        &self,
        v1: &[f64],
        v2: &[f64],
        v1_binary: bool,
        v2_binary: bool,
    ) -> Result<f64, Error> {
        if v1.len() != v2.len() {
            return Err(Error::VectorLengthMismatch);
        }

        if v1.is_empty() {
            return Err(Error::VectorEmpty);
        }

        let result = match (v1_binary, v2_binary) {
            (true, true) => dist_corr_both_binary(v1, v2),
            (true, false) => {
                if !v1.iter().all(|&x| x == 0.0 || x == 1.0) {
                    return Err(Error::VectorNotBinary);
                } else {
                    dist_corr_one_binary(v1, v2)
                }
            }
            (false, true) => {
                if !v2.iter().all(|&x| x == 0.0 || x == 1.0) {
                    return Err(Error::VectorNotBinary);
                } else {
                    dist_corr_one_binary(v2, v1)
                }
            }
            (false, false) => dist_corr(v1, v2),
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
    /// # Errors
    ///
    /// The function will return an error if:
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use dist_corr::DistCovariance;
    ///
    /// let v1 = vec![0.0, 1.0, 0.0, 1.0];
    /// let v2 = vec![0.0, 1.0, 1.0, 0.0];
    ///
    /// let dist_cov = DistCovariance;
    /// let result = dist_cov.compute(&v1, &v2).unwrap();
    ///
    /// assert_eq!(result, 0.0);
    /// ```
    pub fn compute(&self, v1: &[f64], v2: &[f64]) -> Result<f64, Error> {
        self.compute_binary(v1, v2, false, false)
    }

    /// Computes the distance covariance between two vectors where at least one is binary, i.e. 0-1-valued.
    ///
    /// # Arguments
    ///
    /// * `v1` - A slice of `f64` values representing the first data vector.
    /// * `v2` - A slice of `f64` values representing the second data vector.
    /// * `v1_binary` - A flag indicating if v1 is a binary vector, i.e. with values either 0.0 or 1.0.
    /// * `v2_binary` - A flag indicating if v2 is a binary vector, i.e. with values either 0.0 or 1.0.
    ///
    /// # Returns
    ///
    /// Returns a `f64` representing the distance covariance between the two binary input vectors.
    ///
    /// # Errors
    ///
    /// The function will return an error if:
    /// - `v1` or `v2` is not 0-1-valued as indicated by v1_binary and v2_binary
    /// - The lengths of `v1` and `v2` do not match.
    /// - Either of the vectors is empty.
    ///
    /// ```
    /// use dist_corr::DistCovariance;
    ///
    /// let v1 = vec![0.0, 1.0, 0.0, 1.0];
    /// let v2 = vec![0.0, 1.0, 1.0, 0.0];
    ///
    /// let dist_cov = DistCovariance;
    /// let result = dist_cov.compute_binary(&v1, &v2, true, true).unwrap();
    ///
    /// assert_eq!(result, 0.0);
    /// ```
    pub fn compute_binary(
        &self,
        v1: &[f64],
        v2: &[f64],
        v1_binary: bool,
        v2_binary: bool,
    ) -> Result<f64, Error> {
        if v1.len() != v2.len() {
            return Err(Error::VectorLengthMismatch);
        }

        if v1.is_empty() {
            return Err(Error::VectorEmpty);
        }

        match (v1_binary, v2_binary) {
            (true, true) => dist_cov_both_binary(v1, v2),
            (true, false) => {
                if v1_binary && !v1.iter().all(|&x| x == 0.0 || x == 1.0) {
                    return Err(Error::VectorNotBinary);
                };
                dist_cov_one_binary(v1, v2)
            }
            (false, true) => {
                if v2_binary && !v2.iter().all(|&x| x == 0.0 || x == 1.0) {
                    return Err(Error::VectorNotBinary);
                };
                dist_cov_one_binary(v2, v1)
            }
            (false, false) => dist_cov(v1, v2),
        }
    }

    /// Computes the distance variance of a single vector.
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
    ///
    /// This method is faster than calling `DistCovariance::compute(v,v)` if `v` is non-binary.
    ///
    /// # Errors
    ///
    /// The function will return an error if:
    /// - The input vector `v` is empty.
    ///
    /// ```
    /// use dist_corr::DistCovariance;
    ///
    /// let v = vec![1.0, 1.0, 1.0];
    ///
    /// let dist_cov = DistCovariance;
    /// let result = dist_cov.compute_var(&v).unwrap();
    ///
    /// assert_eq!(result, 0.0);
    /// ```
    pub fn compute_var(&self, v: &[f64]) -> Result<f64, Error> {
        if v.is_empty() {
            return Err(Error::VectorEmpty);
        }

        Ok(dist_var(v))
    }
}
