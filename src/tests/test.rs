// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use crate::api::{DistCorrelation, DistCovariance};
use crate::dist_corr_naive::_dist_cov_naive;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Tests

#[test]
fn independent() {
    let v1: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
    let v2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let dist_correlation = DistCorrelation;
    let dist_covariance = DistCovariance;

    let tick = Instant::now();
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    let time_fast = tick.elapsed().as_secs_f32();
    let dist_cov = dist_covariance.compute(&v1, &v2).unwrap();

    let tick = Instant::now();
    let dist_cov_naive = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naive = (dist_cov_naive / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();
    let time_naive = tick.elapsed().as_secs_f32();

    println!("Dist corr fast: Time {}s", time_fast);
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: Time {}s", time_naive);
    println!("Dist corr naive: {:?}", dist_corr_naive);

    assert!(dist_corr < f64::EPSILON);
    assert!((dist_corr_naive - dist_corr).abs() < 1e-10);
    assert!((dist_cov - dist_cov_naive).abs() < 1e-10);
}

#[test]
fn independent_2() {
    let sample_size = 1000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .collect();

    let v2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-10.0..10.0))
        .collect();

    let dist_correlation = DistCorrelation;
    let dist_covariance = DistCovariance;

    let tick = Instant::now();
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    let time_fast = tick.elapsed().as_secs_f32();
    let dist_cov = dist_covariance.compute(&v1, &v2).unwrap();

    let tick = Instant::now();
    let dist_cov_naive = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naive = (dist_cov_naive / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();
    let time_naive = tick.elapsed().as_secs_f32();

    println!("Dist corr fast: Time {}s", time_fast);
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: Time {}s", time_naive);
    println!("Dist corr naive: {:?}", dist_corr_naive);

    assert!((dist_corr_naive - dist_corr).abs() < 1e-10);
    assert!((dist_cov - dist_cov_naive).abs() < 1e-10);
}

#[test]
/// the solution is sqrt(2/sqrt(40)) ~ 0.56234132519
fn quadratic_relation_simple() {
    let v1: Vec<f64> = vec![1.0, 0.0, -1.0];
    let v2: Vec<f64> = v1.iter().map(|x| x * x).collect();

    let dist_correlation = DistCorrelation;
    let dist_covariance = DistCovariance;

    let tick = Instant::now();
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    let time_fast = tick.elapsed().as_secs_f32();
    let dist_cov = dist_covariance.compute(&v1, &v2).unwrap();

    let exact_solution = (2.0_f64 / 40.0_f64.sqrt()).sqrt();
    assert!((dist_corr - exact_solution).abs() < 1e-10);

    let tick = Instant::now();
    let dist_cov_naive = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naive = (dist_cov_naive / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();
    let time_naive = tick.elapsed().as_secs_f32();

    println!("Exact solution: {:?}", exact_solution);
    println!("Dist corr fast: Time {}s", time_fast);
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: Time {}s", time_naive);
    println!("Dist corr naive: {:?}", dist_corr_naive);

    assert!((dist_corr - exact_solution).abs() < 1e-10);
    assert!((dist_cov - dist_cov_naive).abs() < 1e-10);
}

#[test]
fn quadratic_relation() {
    let test_sizes = [2_i32.pow(4), 2_i32.pow(10), 111, 897];

    for numb in test_sizes {
        sub_test(numb as usize, 13, |x| x * x);
    }
}

#[test]
fn sin() {
    let test_sizes = [2_i32.pow(6), 2_i32.pow(11), 121, 597];

    for numb in test_sizes {
        sub_test(numb as usize, 21, |x| x.sin());
    }
}

#[test]
fn linear_relation() {
    let test_sizes = [2_i32.pow(14), 2_i32.pow(9), 131, 577];

    for numb in test_sizes {
        sub_test(numb as usize, 21, |x| x * 0.000001 - 0.3);
    }
}

fn sub_test(sample_size: usize, seed: u64, func: fn(&f64) -> f64) {
    println!("------------------------");
    println!("Sample size: {:?}", sample_size);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();

    let v2: Vec<f64> = v1.iter().map(func).collect();

    let dist_correlation = DistCorrelation;
    let dist_covariance = DistCovariance;

    let tick = Instant::now();
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    let time_fast = tick.elapsed().as_secs_f32();
    let dist_cov = dist_covariance.compute(&v1, &v2).unwrap();
    println!("Dist corr fast: Time {}s", time_fast);
    println!("Dist corr: {:?}", dist_corr);

    let tick = Instant::now();
    let dist_cov_naive = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naive = (dist_cov_naive / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();
    let time_naive = tick.elapsed().as_secs_f32();
    println!("Dist corr naive: Time {}s", time_naive);
    println!("Dist corr naive: {:?}", dist_corr_naive);

    assert!((dist_corr_naive - dist_corr).abs() < 1e-5);
    assert!((dist_cov - dist_cov_naive) < 1e-10);
}
