use crate::api::{dist_corr_binary, dist_corr_one_binary};
use crate::dist_corr_binary::{
    dist_corr_fast_one_binary, dist_cov_binary_sqrt, dist_cov_one_binary,
};
use crate::dist_corr_fast::{dist_corr_fast, dist_cov_fast};
use crate::dist_corr_naive::_dist_cov_naive;

use std::time::Instant;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[test]
fn simple_binary() {
    let v_1: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
    let v_2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v_1, &v_2);

    assert!(dist_corr < f64::EPSILON);

    let dist_cov_naiv = _dist_cov_naive(&v_1, &v_2);
    let dist_var_v_1 = _dist_cov_naive(&v_1, &v_1);
    let dist_var_v_2 = _dist_cov_naive(&v_2, &v_2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v_1 * dist_var_v_2).sqrt()).sqrt();

    assert!(dist_corr_naiv < f64::EPSILON);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_corr_binary = (dist_cov_binary_sqrt(&v_1, &v_2)
        / (dist_cov_binary_sqrt(&v_1, &v_1) * dist_cov_binary_sqrt(&v_2, &v_2)).sqrt())
    .sqrt();
    println!("Dist corr binary: {:?}", dist_corr_binary);
}

#[test]
fn medium_binary() {
    let sample_size = 20000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v_1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let v_2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v_1, &v_2);

    println!("Time dist corr fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);

    let tick = Instant::now();
    let dist_corr_binary = dist_cov_binary_sqrt(&v_1, &v_2)
        / (dist_cov_binary_sqrt(&v_1, &v_1) * dist_cov_binary_sqrt(&v_2, &v_2)).sqrt();

    println!("Time dist corr binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr binary: {:?}", dist_corr_binary);
}

#[test]
fn simple_one_binary() {
    let v_1: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0];
    let v_2: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v_1, &v_2);

    assert!(dist_corr < f64::EPSILON);

    let dist_cov_naiv = _dist_cov_naive(&v_1, &v_2);
    let dist_var_v_1 = _dist_cov_naive(&v_1, &v_1);
    let dist_var_v_2 = _dist_cov_naive(&v_2, &v_2);
    let dist_corr_naiv = (dist_cov_naiv / (dist_var_v_1 * dist_var_v_2).sqrt()).sqrt();

    assert!(dist_corr_naiv < f64::EPSILON);

    println!("Time {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);
    println!("Dist corr naive: {:?}", dist_corr_naiv);

    let dist_corr_binary = dist_corr_fast_one_binary(&v_1, &v_2);
    println!("Dist corr binary: {:?}", dist_corr_binary);
}

#[test]
fn medium_one_binary() {
    let sample_size = 20000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v_1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let v_2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let tick = Instant::now();

    let dist_corr = dist_corr_fast(&v_1, &v_2);

    println!("Time dist corr fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);

    let tick = Instant::now();

    let dist_corr_binary = dist_corr_fast_one_binary(&v_1, &v_2);

    println!("Time dist corr binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr binary: {:?}", dist_corr_binary);
}

#[test]
fn hard_one_binary() {
    let sample_size = 100000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v_1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let v_2: Vec<f64> = (0..sample_size)
        .map(move |_x| rng_2.gen_range(-1.0..1.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let tick = Instant::now();
    let dist_corr = dist_corr_fast(&v_1, &v_2);
    println!("Time dist corr fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr: {:?}", dist_corr);

    let tick = Instant::now();
    let dist_corr_one_binary = dist_corr_one_binary(&v_1, &v_2);
    println!(
        "Time dist corr one binary {}s",
        tick.elapsed().as_secs_f32()
    );
    println!("Dist corr one binary: {:?}", dist_corr_one_binary);

    let tick = Instant::now();
    let dist_corr_binary = dist_corr_binary(&v_1, &v_2);
    println!("Time dist corr binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist corr binary: {:?}", dist_corr_binary);

    println!("-------------------------");
    println!("Dist Cov");
    println!("-------------------------");

    let tick = Instant::now();
    let dist_cov_fast = dist_cov_fast(&v_1, &v_2);
    println!("Time dist cov fast {}s", tick.elapsed().as_secs_f32());
    println!("Dist cov fast: {:?}", dist_cov_fast);

    let tick = Instant::now();
    let dist_cov_one_binary = dist_cov_one_binary(&v_1, &v_2);
    println!("Time dist cov one binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist cov one binary: {:?}", dist_cov_one_binary);

    println!("-------------------------");
    println!("-------------------------");

    let tick = Instant::now();
    let dist_cov_binary = dist_cov_binary_sqrt(&v_1, &v_2);

    println!("Time dist cov binary {}s", tick.elapsed().as_secs_f32());
    println!("Dist cov binary: {:?}", (dist_cov_binary).powi(2));
}
