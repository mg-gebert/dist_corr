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
