// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use crate::{DistCorrelation, DistCovariance};

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Tests

#[test]
fn covariance_var() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result_seq = DistCovariance::new(false).compute_var(&v).unwrap();
    let result_par = DistCovariance::new(true).compute_var(&v).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn covariance_cov() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let result_seq = DistCovariance::new(false).compute(&v1, &v2).unwrap();
    let result_par = DistCovariance::new(true).compute(&v1, &v2).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn covariance_cov_binary() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let result_seq = DistCovariance::new(false)
        .compute_binary(&v1, &v2, false, true)
        .unwrap();
    let result_par = DistCovariance::new(true)
        .compute_binary(&v1, &v2, false, true)
        .unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn covariance_cov_binary_both() {
    let v1 = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let result_seq = DistCovariance::new(false)
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    let result_par = DistCovariance::new(true)
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn correlation() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let result_seq = DistCorrelation::new(false).compute(&v1, &v2).unwrap();
    let result_par = DistCorrelation::new(true).compute(&v1, &v2).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn correlation_binary() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let result_seq = DistCorrelation::new(false)
        .compute_binary(&v1, &v2, false, true)
        .unwrap();
    let result_par = DistCorrelation::new(true)
        .compute_binary(&v1, &v2, false, true)
        .unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn correlation_binary_both() {
    let v1 = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let result_seq = DistCorrelation::new(false)
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    let result_par = DistCorrelation::new(true)
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn correlation_large() {
    // large enough that the parallel path splits the work into several chunks
    let n = 20_000;
    let v1: Vec<f64> = (0..n).map(|j| (j as f64).sin()).collect();
    let v2: Vec<f64> = (0..n).map(|j| (j as f64).cos()).collect();
    let result_seq = DistCorrelation::new(false).compute(&v1, &v2).unwrap();
    let result_par = DistCorrelation::new(true).compute(&v1, &v2).unwrap();
    assert!(f64::abs(result_seq - result_par) < 1e-10);
}
