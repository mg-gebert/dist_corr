use crate::dist_corr_binary::dist_cov_one_binary as dist_cov_one_binary_other;
use crate::dist_corr_binary::{
    dist_corr_fast_binary, dist_corr_fast_one_binary, dist_cov_binary_sqrt,
};
use crate::dist_corr_fast::{dist_corr_fast, dist_cov_fast, dist_var_fast};

/// Computes the distance correlation between two vectors.
///
/// # Arguments
///
/// * `v_1` - A slice of `f64` values representing the first data vector.
/// * `v_2` - A slice of `f64` values representing the second data vector.
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
/// - The lengths of `v_1` and `v_2` do not match.
/// - Either of the vectors is empty.
///
/// # Examples
///
/// ```
/// use dist_corr::dist_corr;
///
/// let v_1 = [1.0, 0.0, -1.0];
/// let v_2 = [1.0, 0.0, 1.0];
/// let corr = dist_corr(&v_1, &v_2);
///
/// println!("Distance correlation: {}", corr);
/// ```
pub fn dist_corr(v_1: &[f64], v_2: &[f64]) -> f64 {
    dist_corr_fast(v_1, v_2)
}

/// Computes the distance correlation between two binary (0-1) vectors.
///
/// # Arguments
///
/// * `v_1` - A float valued slice with values either 0.0 or 1.0 representing the first data vector.
/// * `v_2` - A float valued slice with values either 0.0 or 1.0 representing the second data vector.
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
/// - `v_1` or `v_2` is not 0.0 - 1.0 valued
/// - The lengths of `v_1` and `v_2` do not match.
/// - Either of the vectors is empty.
///
/// # Examples
///
/// ```
/// use dist_corr::dist_corr;
///
/// let v_1 = [1.0, 0.0, 0.0];
/// let v_2 = [1.0, 0.0, 1.0];
/// let corr = dist_corr_binary(&v_1, &v_2);
///
/// println!("Distance correlation: {}", corr);
/// ```
pub fn dist_corr_binary(v_1: &[f64], v_2: &[f64]) -> f64 {
    // check if v_1 and v_2 are binary
    assert!(
        v_1.iter().all(|&x| (x == 0.0 || x == 1.0)),
        "v_1 must contain only 0.0 or 1.0 values"
    );
    assert!(
        v_2.iter().all(|&x| x == 0.0 || x == 1.0),
        "v_2 must contain only 0.0 or 1.0 values"
    );

    dist_corr_fast_binary(v_1, v_2)
}

/// Computes the distance correlation between a binary (0-1) vector and a float valued vector.
///
/// # Arguments
///
/// * `v_1` - A float valued slice with values either 0.0 or 1.0 representing the first data vector.
/// * `v_2` - A slice of `f64` values representing the second data vector.
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
/// - `v_1` is not binary, i.e. 0.0-1.0 valued
/// - The lengths of `v_1` and `v_2` do not match.
/// - Either of the vectors is empty.
///
/// # Examples
///
/// ```
/// use dist_corr::dist_corr_one_binary;
///
/// let v_1 = [1.0, 0.0, 0.0];
/// let v_2 = [1.0, 2.1, -1.1];
/// let corr = dist_corr_one_binary(&v_1, &v_2);
///
/// println!("Distance correlation: {}", corr);
/// ```
pub fn dist_corr_one_binary(v_1: &[f64], v_2: &[f64]) -> f64 {
    // check if v_1 is binary
    /*
    assert!(
        v_1.iter().all(|&x| x == 0.0 || x == 1.0),
        "v_1 must contain only 0.0 or 1.0 values"
    );
    */
    dist_corr_fast_one_binary(v_1, v_2)
}

/// Computes the distance covariance between two vectors.
///
/// Distance covariance is a measure of dependence between two random variables.
///
/// # Arguments
///
/// * `v_1` - A slice of `f64` values representing the first data vector.
/// * `v_2` - A slice of `f64` values representing the second data vector.
///
/// # Returns
///
/// A `f64` representing the distance covariance between the two input vectors.
/// The result is always non-negative, where:
/// - `0.0` indicates independence under certain conditions.
///
/// # Panics
///
/// The function will panic if:
/// - The lengths of `v_1` and `v_2` do not match.
/// - Either of the vectors is empty.
///
/// # Examples
///
/// ```
/// use dist_corr::dist_cov;
///
/// let v_1 = vec![1.0, 0.0, -1.0];
/// let v_2 = vec![1.0, 0.0, 1.0];
/// let cov = dist_cov(&v_1, &v_2);
///
/// println!("Distance covariance: {}", cov);
/// ```
pub fn dist_cov(v_1: &[f64], v_2: &[f64]) -> f64 {
    dist_cov_fast(v_1, v_2)
}

/// Computes the distance covariance between  between two binary (0-1) vectors.
///
/// Distance covariance is a measure of dependence between two random variables.
///
/// # Arguments
///
/// * `v_1` - A float valued slice with values either 0.0 or 1.0 representing the first data vector.
/// * `v_2` - A slice of `f64` values representing the second data vector.
///
/// # Returns
///
/// A `f64` representing the distance covariance between the two input vectors.
/// The result is always non-negative, where:
/// - `0.0` indicates independence under certain conditions.
///
/// # Panics
///
/// The function will panic if:
/// - `v_1` or `v_2` is not 0.0 - 1.0 valued
/// - The lengths of `v_1` and `v_2` do not match.
/// - Either of the vectors is empty.
///
/// # Examples
///
/// ```
/// use dist_corr::dist_cov_binary;
///
/// let v_1 = vec![1.0, 0.0, -1.0];
/// let v_2 = vec![1.0, 0.0, 1.0];
/// let cov = dist_cov_binary(&v_1, &v_2);
///
/// println!("Distance covariance: {}", cov);
/// ```
pub fn dist_cov_binary(v_1: &[f64], v_2: &[f64]) -> f64 {
    dist_cov_binary_sqrt(v_1, v_2).powi(2)
}

/// Computes the distance covariance between a binary (0-1) vector and a float valued vector.
///
/// # Arguments
///
/// * `v_1` - A float valued slice with values either 0.0 or 1.0 representing the first data vector.
/// * `v_2` - A slice of `f64` values representing the second data vector.
///
/// # Returns
///
/// A `f64` representing the distance covariance between the two input vectors.
/// The result is always non-negative, where:
/// - `0.0` indicates independence under certain conditions.
///
/// # Panics
///
/// The function will panic if:
/// - `v_1` is not binary, i.e. 0.0-1.0 valued
/// - The lengths of `v_1` and `v_2` do not match.
/// - Either of the vectors is empty.
///
/// # Examples
///
/// ```
/// use dist_corr::dist_cov_one_binary;
///
/// let v_1 = vec![1.0, 0.0, -1.0];
/// let v_2 = vec![1.0, 0.0, 1.0];
/// let cov = dist_cov_one_binary(&v_1, &v_2);
///
/// println!("Distance covariance: {}", cov);
/// ```
pub fn dist_cov_one_binary(v_1: &[f64], v_2: &[f64]) -> f64 {
    dist_cov_one_binary_other(v_1, v_2)
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
/// # Panics
///
/// This function will panic if:
/// - The input vector `v` is empty.
///
/// # Examples
///
/// ```
/// use dist_corr::dist_var;
///
/// let v = [1.0, 0.0, -1.0];
/// let var = dist_var(&v);
///
/// println!("Distance variance: {}", var);
/// ```
pub fn dist_var(v: &[f64]) -> f64 {
    dist_var_fast(v)
}
