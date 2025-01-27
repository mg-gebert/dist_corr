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
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let v_1: Vec<f64> = (0..2_i32.pow(10) as usize)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();

    let v_2: Vec<f64> = v_1.iter().map(|x| x * x).collect();

    let mut dist_corr = 0.0;

    let tick = Instant::now();
    for _i in 0..1 {
        dist_corr = dist_corr_fast(&v_1, &v_2);
    }
    println!("Time {}s", tick.elapsed().as_secs_f32() / 1.0);
    println!("Dist corr: {:?}", dist_corr);

    let tick = Instant::now();
    let a = dist_cov_naive(&v_1, &v_2);
    let b = dist_cov_naive(&v_1, &v_1);
    let c = dist_cov_naive(&v_2, &v_2);

    println!("Time {}s", tick.elapsed().as_secs_f32() / 1.0);

    println!("Dist corr naive: {:?}", (a / (b * c).sqrt()).sqrt());

    let dist_exp_cov = dist_cov_naive_exp(&v_1, &v_2);
    let dist_exp_var_v_1 = dist_cov_naive_exp(&v_1, &v_1);
    let dist_exp_var_v_2 = dist_cov_naive_exp(&v_2, &v_2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v_1 * dist_exp_var_v_2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
}

#[test]
fn linear_relation() {
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let v_1: Vec<f64> = (0..16000)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();

    //let mut v_1_sorted = v_1.to_vec();
    //v_1_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let v_2: Vec<f64> = v_1.iter().map(|x| 0.1 * x + 0.2).collect();

    let mut dist_corr = 0.0;

    let tick = Instant::now();
    for _i in 0..100 {
        dist_corr = dist_corr_fast(&v_1, &v_2);
    }
    println!("Time {}s", tick.elapsed().as_secs_f32() / 100.0);

    /*
        let tick = Instant::now();
        let a = dist_cov_naive(&v_2, &v_2);

        println!("Time {}s", tick.elapsed().as_secs_f32() / 1.0);
    */
    println!("Dist corr: {:?}", dist_corr);

    let dist_exp_cov = dist_cov_naive_exp(&v_1, &v_2);
    let dist_exp_var_v_1 = dist_cov_naive_exp(&v_1, &v_1);
    let dist_exp_var_v_2 = dist_cov_naive_exp(&v_2, &v_2);
    let dist_exp_corr = (dist_exp_cov / (dist_exp_var_v_1 * dist_exp_var_v_2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_exp_corr);
}
