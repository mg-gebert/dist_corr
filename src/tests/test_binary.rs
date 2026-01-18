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
fn simple_binary() {
    let v1: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
    let v2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let dist_correlation = DistCorrelation;

    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    println!("Dist corr: {:?}", dist_corr);
    assert!(dist_corr < f64::EPSILON);

    let dist_cov_naive = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naive = (dist_cov_naive / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();
    println!("Dist corr naive: {:?}", dist_corr_naive);
    assert!(dist_corr_naive < f64::EPSILON);

    let dist_corr_binary = dist_correlation
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    println!("Dist corr binary: {:?}", dist_corr_binary);
    assert!(dist_corr_binary < f64::EPSILON);
}

#[test]
fn medium_binary() {
    let sample_size = 20000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let v2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let dist_correlation = DistCorrelation;
    let dist_covariance = DistCovariance;

    let tick = Instant::now();
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    println!("Time dist corr fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    let dist_cov = dist_covariance.compute(&v1, &v2).unwrap();

    let dist_cov_naive = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naive = (dist_cov_naive / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();
    println!("Dist corr naive: {:?}", dist_corr_naive);

    let tick = Instant::now();
    let dist_corr_binary = dist_correlation
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    println!("Time dist corr binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr binary: {:?}", dist_corr_binary);
    let dist_cov_binary = dist_covariance
        .compute_binary(&v1, &v2, true, true)
        .unwrap();

    assert!((dist_corr_binary - dist_corr).abs() < 1e-10);
    assert!((dist_cov - dist_cov_binary).abs() < 1e-10);
}

#[test]
fn simple_one_binary() {
    let v1: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0];
    let v2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let tick = Instant::now();
    let dist_correlation = DistCorrelation;
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    assert!(dist_corr < f64::EPSILON);

    let dist_cov_naive = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naive = (dist_cov_naive / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();

    assert!(dist_corr_naive < f64::EPSILON);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naive);

    let dist_corr_binary = dist_correlation
        .compute_binary(&v1, &v2, true, false)
        .unwrap();
    println!("Dist corr binary: {:?}", dist_corr_binary);

    assert!(dist_corr_binary.abs() < f64::EPSILON);
}

#[test]
fn medium_one_binary() {
    let sample_size = 20000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let v2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let dist_correlation = DistCorrelation;
    let dist_covariance = DistCovariance;

    let tick = Instant::now();
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    println!("Time dist corr fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    let dist_cov = dist_covariance.compute(&v1, &v2).unwrap();

    let tick = Instant::now();
    let dist_corr_binary = dist_correlation
        .compute_binary(&v1, &v2, true, false)
        .unwrap();
    println!("Time dist corr binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr binary: {:?}", dist_corr_binary);
    let dist_cov_binary = dist_covariance
        .compute_binary(&v1, &v2, true, false)
        .unwrap();

    assert!((dist_corr - dist_corr_binary).abs() < 1e-10);
    assert!((dist_cov - dist_cov_binary).abs() < 1e-10);
}

#[test]
fn hard_one_binary() {
    let sample_size = 2_i32.pow(22); //1_000_000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(76);

    println!("Length of vectors: {:?}", sample_size);

    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let v2: Vec<f64> = v1
        .iter()
        .enumerate()
        .map(|(i, x)| if i % 2 == 0 { *x } else { 0.0 })
        .collect();

    let dist_correlation = DistCorrelation;

    println!("-------------------------");
    println!("Dist Corr with Frobenius");
    let tick = Instant::now();
    let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
    println!("Result: {:?}", dist_corr);
    println!("Time {}s", tick.elapsed().as_secs_f32());

    println!("-------------------------");
    println!("Dist Corr with one binary");
    let tick = Instant::now();
    let dist_corr_one_binary = dist_correlation
        .compute_binary(&v1, &v2, true, false)
        .unwrap();
    println!("Result: {:?}", dist_corr_one_binary);
    println!("Time {}s", tick.elapsed().as_secs_f32());
    let dist_corr_one_binary_flipped = dist_correlation
        .compute_binary(&v2, &v1, false, true)
        .unwrap();

    println!("-------------------------");
    println!("Dist Corr with two binary");
    let tick = Instant::now();
    let dist_corr_binary = dist_correlation
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    println!("Result: {:?}", dist_corr_binary);
    println!("Time {}s", tick.elapsed().as_secs_f32());

    assert!((dist_corr_one_binary_flipped - dist_corr_one_binary).abs() < 1e-20);
    assert!((dist_corr - dist_corr_one_binary).abs() < 1e-10);
    assert!((dist_corr_one_binary - dist_corr_binary).abs() < 1e-10);

    let distance_covariance = DistCovariance;
    let dist_cov = distance_covariance.compute(&v1, &v2).unwrap();
    println!("dist_cov: {:?}", dist_cov);

    let dist_cov_binary = distance_covariance
        .compute_binary(&v1, &v2, true, true)
        .unwrap();
    println!("dist_cov_binary: {:?}", dist_cov_binary);

    assert!((dist_cov - dist_cov_binary).abs() < 1e-10);
}
