use crate::{DistCovariance, DistCorrelation};

#[test]
fn covariance_var() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let dist_cov = DistCovariance;
    let result_seq = dist_cov.compute_var(&v).unwrap();
    let result_par = dist_cov.par_compute_var(&v).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn covariance_cov() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let dist_cov = DistCovariance;
    let result_seq = dist_cov.compute(&v1, &v2).unwrap();
    let result_par = dist_cov.par_compute(&v1, &v2).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn covariance_cov_binary() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let dist_cov = DistCovariance;
    let result_seq = dist_cov.compute_binary(&v1, &v2, false, true).unwrap();
    let result_par = dist_cov.par_compute_binary(&v1, &v2, false, true).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn covariance_cov_binary_both() {
    let v1 = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let dist_cov = DistCovariance;
    let result_seq = dist_cov.compute_binary(&v1, &v2, true, true).unwrap();
    let result_par = dist_cov.par_compute_binary(&v1, &v2, true, true).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn correlation() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let dist_corr = DistCorrelation;
    let result_seq = dist_corr.compute(&v1, &v2).unwrap();
    let result_par = dist_corr.par_compute(&v1, &v2).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn correlation_binary() {
    let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let dist_corr = DistCorrelation;
    let result_seq = dist_corr.compute_binary(&v1, &v2, false, true).unwrap();
    let result_par = dist_corr.par_compute_binary(&v1, &v2, false, true).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}

#[test]
fn correlation_binary_both() {
    let v1 = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let v2 = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let dist_corr = DistCorrelation;
    let result_seq = dist_corr.compute_binary(&v1, &v2, true, true).unwrap();
    let result_par = dist_corr.par_compute_binary(&v1, &v2, true, true).unwrap();
    assert!(f64::abs(result_seq - result_par) < f64::EPSILON);
}