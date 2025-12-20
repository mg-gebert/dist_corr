// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use crate::api::dist_cov_binary;
use crate::dist_corr_binary::{dist_corr_fast_binary, dist_corr_fast_one_binary};
use crate::dist_corr_fast::dist_corr_fast;
use crate::dist_corr_fast::dist_cov_fast;
use crate::dist_corr_naive::_dist_cov_naive;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Tests

#[test]
fn simple_binary() {
    let v1: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
    let v2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v1, &v2).unwrap();

    assert!(dist_corr < f64::EPSILON);

    let dist_cov_naiv = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();

    assert!(dist_corr_naiv < f64::EPSILON);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_corr_binary = (dist_cov_binary(&v1, &v2).unwrap()
        / (dist_cov_binary(&v1, &v1).unwrap() * dist_cov_binary(&v2, &v2).unwrap()).sqrt())
    .sqrt();
    println!("Dist corr binary: {:?}", dist_corr_binary);
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

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v1, &v2);

    println!("Time dist corr fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);

    let tick = Instant::now();
    let dist_corr_binary = dist_cov_binary(&v1, &v2).unwrap()
        / (dist_cov_binary(&v1, &v1).unwrap() * dist_cov_binary(&v2, &v2).unwrap()).sqrt();

    println!("Time dist corr binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr binary: {:?}", dist_corr_binary);
}

#[test]
fn simple_one_binary() {
    let v1: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0];
    let v2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v1, &v2).unwrap();

    assert!(dist_corr < f64::EPSILON);

    let dist_cov_naiv = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();

    assert!(dist_corr_naiv < f64::EPSILON);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_corr_binary = dist_corr_fast_one_binary(&v1, &v2);
    println!("Dist corr binary: {:?}", dist_corr_binary);
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

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v1, &v2);

    println!("Time dist corr fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);

    let tick = Instant::now();

    let dist_corr_binary = dist_corr_fast_one_binary(&v1, &v2);

    println!("Time dist corr binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr binary: {:?}", dist_corr_binary);
}

#[test]
fn hard_one_binary() {
    let sample_size = 1000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);

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

    println!("-------------------------");
    println!("Dist Corr with Frobenius");
    let tick = Instant::now();
    let dist_corr = dist_corr_fast(&v1, &v2);
    println!("Result: {:?}", dist_corr);
    println!("Time {}s", tick.elapsed().as_secs_f32());

    println!("-------------------------");
    println!("Dist Corr with one binary");
    let tick = Instant::now();
    let dist_corr_one_binary = dist_corr_fast_one_binary(&v1, &v2);
    println!("Result: {:?}", dist_corr_one_binary);
    println!("Time {}s", tick.elapsed().as_secs_f32());

    println!("-------------------------");
    println!("Dist Corr with two binary");
    let tick = Instant::now();
    let dist_corr_binary = dist_corr_fast_binary(&v1, &v2);
    println!("Result: {:?}", dist_corr_binary);
    println!("Time {}s", tick.elapsed().as_secs_f32());

    let dist_cov = dist_cov_fast(&v1, &v2);
    println!("dist_cov: {:?}", dist_cov);

    let dist_cov_binary = dist_cov_binary(&v1, &v2);

    println!("dist_cov_binary: {:?}", dist_cov_binary);
}
