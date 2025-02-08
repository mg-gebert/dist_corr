use crate::dist_corr_fast::dist_corr_fast;
use crate::dist_corr_naive::{dist_cov_naive, dist_cov_naive_exp};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

#[test]
fn independent() {
    let v_1: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
    let v_2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v_1, &v_2);

    assert!(dist_corr < f64::EPSILON);

    let dist_cov_naiv = dist_cov_naive(&v_1, &v_2);
    let dist_var_v_1 = dist_cov_naive(&v_1, &v_1);
    let dist_var_v_2 = dist_cov_naive(&v_2, &v_2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v_1 * dist_var_v_2).sqrt()).sqrt();

    assert!(dist_corr_naiv < f64::EPSILON);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_exp_cov = dist_cov_naive_exp(&v_1, &v_2);
    let dist_exp_var_v_1 = dist_cov_naive_exp(&v_1, &v_1);
    let dist_exp_var_v_2 = dist_cov_naive_exp(&v_2, &v_2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v_1 * dist_exp_var_v_2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
}

#[test]
/// the solution is sqrt(2/sqrt(40)) ~ 0.56234132519
fn quadratic_relation_simple() {
    let v_1: Vec<f64> = vec![1.0, 0.0, -1.0];
    let v_2: Vec<f64> = v_1.iter().map(|x| x * x).collect();

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v_1, &v_2);

    let solution = (2.0_f64 / 40.0_f64.sqrt()).sqrt();
    assert!((dist_corr - solution).abs() < 1e-10);

    let dist_cov_naiv = dist_cov_naive(&v_1, &v_2);
    let dist_var_v_1 = dist_cov_naive(&v_1, &v_1);
    let dist_var_v_2 = dist_cov_naive(&v_2, &v_2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v_1 * dist_var_v_2).sqrt()).sqrt();
    assert!((dist_corr - solution).abs() < 1e-10);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_exp_cov = dist_cov_naive_exp(&v_1, &v_2);
    let dist_exp_var_v_1 = dist_cov_naive_exp(&v_1, &v_1);
    let dist_exp_var_v_2 = dist_cov_naive_exp(&v_2, &v_2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v_1 * dist_exp_var_v_2).sqrt()).sqrt();

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
    let test_sizes = [2_i32.pow(7), 2_i32.pow(9), 131, 577];

    for numb in test_sizes {
        sub_test(numb as usize, 21, |x| x * 0.1 - 0.2);
    }
}

fn sub_test(sample_size: usize, seed: u64, func: fn(&f64) -> f64) {
    println!("------------------------");
    println!("Sample size: {:?}", sample_size);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v_1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();

    let v_2: Vec<f64> = v_1.iter().map(func).collect();

    let dist_corr = dist_corr_fast(&v_1, &v_2);
    println!("Dist corr: {:?}", dist_corr);

    let a = dist_cov_naive(&v_1, &v_2);
    let b = dist_cov_naive(&v_1, &v_1);
    let c = dist_cov_naive(&v_2, &v_2);

    let dist_corr_naive = (a / (b * c).sqrt()).sqrt();
    println!("Dist corr naive: {:?}", dist_corr_naive);

    assert!((dist_corr_naive - dist_corr).abs() < 1e-10);

    let dist_exp_cov = dist_cov_naive_exp(&v_1, &v_2);
    let dist_exp_var_v_1 = dist_cov_naive_exp(&v_1, &v_1);
    let dist_exp_var_v_2 = dist_cov_naive_exp(&v_2, &v_2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v_1 * dist_exp_var_v_2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
}
