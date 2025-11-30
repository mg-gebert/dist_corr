use crate::dist_corr_fast::dist_corr_fast;
use crate::dist_corr_naive::{_dist_cov_naive, _dist_cov_naive_exp};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

#[test]
fn independent() {
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

    let dist_exp_cov = _dist_cov_naive_exp(&v1, &v2);
    let dist_exp_var_v1 = _dist_cov_naive_exp(&v1, &v1);
    let dist_exp_var_v2 = _dist_cov_naive_exp(&v2, &v2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v1 * dist_exp_var_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
}

#[test]
fn independent_2() {
    let sample_size = 10000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .collect();

    let v2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-10.0..10.0))
        .collect();

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v1, &v2);

    //assert!(dist_corr < f64::EPSILON);

    let dist_cov_naiv = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();

    //assert!(dist_corr_naiv < f64::EPSILON);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_exp_cov = _dist_cov_naive_exp(&v1, &v2);
    let dist_exp_var_v1 = _dist_cov_naive_exp(&v1, &v1);
    let dist_exp_var_v2 = _dist_cov_naive_exp(&v2, &v2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v1 * dist_exp_var_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
}

#[test]
/// the solution is sqrt(2/sqrt(40)) ~ 0.56234132519
fn quadratic_relation_simple() {
    let v1: Vec<f64> = vec![1.0, 0.0, -1.0];
    let v2: Vec<f64> = v1.iter().map(|x| x * x).collect();

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v1, &v2).unwrap();

    let solution = (2.0_f64 / 40.0_f64.sqrt()).sqrt();
    assert!((dist_corr - solution).abs() < 1e-10);

    let dist_cov_naiv = _dist_cov_naive(&v1, &v2);
    let dist_var_v1 = _dist_cov_naive(&v1, &v1);
    let dist_var_v2 = _dist_cov_naive(&v2, &v2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v1 * dist_var_v2).sqrt()).sqrt();
    assert!((dist_corr - solution).abs() < 1e-10);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_exp_cov = _dist_cov_naive_exp(&v1, &v2);
    let dist_exp_var_v1 = _dist_cov_naive_exp(&v1, &v1);
    let dist_exp_var_v2 = _dist_cov_naive_exp(&v2, &v2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v1 * dist_exp_var_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
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

#[test]
fn linear_relation_2() {
    let sample_size = 10000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .collect();

    let v2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-10.0..10.0))
        .collect();

    let v_3: Vec<f64> = v1
        .iter()
        .zip(v2.iter())
        .map(|(v1, v2)| v1 + 0.3 * v2 + 2.0)
        .collect();

    let dist_corr_1 = dist_corr_fast(&v1, &v_3);
    let dist_corr_2 = dist_corr_fast(&v2, &v_3);
    let dist_corr_12 = dist_corr_fast(&v2, &v1);

    println!("DistCorr 1: {:?}", dist_corr_1);
    println!("DistCorr 2: {:?}", dist_corr_2);
    println!("DistCorr 1-2: {:?}", dist_corr_12);
}

fn sub_test(sample_size: usize, seed: u64, func: fn(&f64) -> f64) {
    println!("------------------------");
    println!("Sample size: {:?}", sample_size);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();

    let v2: Vec<f64> = v1.iter().map(func).collect();

    let dist_corr = dist_corr_fast(&v1, &v2);
    println!("Dist corr: {:?}", dist_corr);

    let a = _dist_cov_naive(&v1, &v2);
    let b = _dist_cov_naive(&v1, &v1);
    let c = _dist_cov_naive(&v2, &v2);

    let dist_corr_naive = (a / (b * c).sqrt()).sqrt();
    println!("Dist corr naive: {:?}", dist_corr_naive);

    //assert!((dist_corr_naive - dist_corr).abs() < 1e-2);

    let dist_exp_cov = _dist_cov_naive_exp(&v1, &v2);
    let dist_exp_var_v1 = _dist_cov_naive_exp(&v1, &v1);
    let dist_exp_var_v2 = _dist_cov_naive_exp(&v2, &v2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v1 * dist_exp_var_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
}
