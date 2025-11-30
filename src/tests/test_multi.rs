use crate::dist_corr_multi::{_dist_cov_multi, _dist_cov_multi_exp};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[test]
#[ignore]
fn simple() {
    let v1: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let v2: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let dist_cov = _dist_cov_multi_exp(&v1, &v2);
    let dist_cov_v1 = _dist_cov_multi_exp(&v1, &v1);
    let dist_cov_v2 = _dist_cov_multi_exp(&v2, &v2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = _dist_cov_multi(&v1, &v2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = _dist_cov_multi(&v1, &v1);
    let dist_cov_stand_v2 = _dist_cov_multi(&v2, &v2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
#[ignore]
fn independent() {
    let v1: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let v2: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
    ];

    let dist_cov = _dist_cov_multi_exp(&v1, &v2);
    let dist_cov_v1 = _dist_cov_multi_exp(&v1, &v1);
    let dist_cov_v2 = _dist_cov_multi_exp(&v2, &v2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = _dist_cov_multi(&v1, &v2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = _dist_cov_multi(&v1, &v1);
    let dist_cov_stand_v2 = _dist_cov_multi(&v2, &v2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
#[ignore]
fn medium() {
    let sample_size = 1000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v1: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_1.gen_range(-1.0..1.0), rng_2.gen_range(-1.0..1.0)])
        .collect();

    let v2: Vec<Vec<f64>> = v1.iter().map(|v1| vec![v1[0] + v1[1]]).collect();

    let dist_cov = _dist_cov_multi_exp(&v1, &v2);
    let dist_cov_v1 = _dist_cov_multi_exp(&v1, &v1);
    let dist_cov_v2 = _dist_cov_multi_exp(&v2, &v2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = _dist_cov_multi(&v1, &v2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = _dist_cov_multi(&v1, &v1);
    let dist_cov_stand_v2 = _dist_cov_multi(&v2, &v2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
#[ignore]
fn independent_medium() {
    let sample_size = 5000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);
    let mut rng_3 = ChaCha8Rng::seed_from_u64(13);

    let v1: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_1.gen_range(-1.0..1.0), rng_2.gen_range(-1.0..1.0)])
        .collect();

    let v2: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_3.gen_range(-1.0..1.0)])
        .collect();

    let dist_cov = _dist_cov_multi_exp(&v1, &v2);
    let dist_cov_v1 = _dist_cov_multi_exp(&v1, &v1);
    let dist_cov_v2 = _dist_cov_multi_exp(&v2, &v2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = _dist_cov_multi(&v1, &v2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = _dist_cov_multi(&v1, &v1);
    let dist_cov_stand_v2 = _dist_cov_multi(&v2, &v2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
#[ignore]
fn independent_medium_2() {
    let sample_size = 10000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v1: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_1.gen_range(-1.0..1.0)])
        .collect();

    let v2: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_2.gen_range(-1.0..1.0)])
        .collect();

    let dist_cov = _dist_cov_multi_exp(&v1, &v2);
    let dist_cov_v1 = _dist_cov_multi_exp(&v1, &v1);
    let dist_cov_v2 = _dist_cov_multi_exp(&v2, &v2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = _dist_cov_multi(&v1, &v2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = _dist_cov_multi(&v1, &v1);
    let dist_cov_stand_v2 = _dist_cov_multi(&v2, &v2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}
