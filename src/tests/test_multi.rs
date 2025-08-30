use crate::dist_corr_multi::{dist_cov_multi, dist_cov_multi_exp};

use std::time::Instant;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[test]
fn simple() {
    let v_1: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let v_2: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let dist_cov = dist_cov_multi_exp(&v_1, &v_2);
    let dist_cov_v1 = dist_cov_multi_exp(&v_1, &v_1);
    let dist_cov_v2 = dist_cov_multi_exp(&v_2, &v_2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = dist_cov_multi(&v_1, &v_2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = dist_cov_multi(&v_1, &v_1);
    let dist_cov_stand_v2 = dist_cov_multi(&v_2, &v_2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
fn independent() {
    let v_1: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let v_2: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
    ];

    let dist_cov = dist_cov_multi_exp(&v_1, &v_2);
    let dist_cov_v1 = dist_cov_multi_exp(&v_1, &v_1);
    let dist_cov_v2 = dist_cov_multi_exp(&v_2, &v_2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = dist_cov_multi(&v_1, &v_2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = dist_cov_multi(&v_1, &v_1);
    let dist_cov_stand_v2 = dist_cov_multi(&v_2, &v_2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
fn medium() {
    let sample_size = 1000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v_1: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_1.gen_range(-1.0..1.0), rng_2.gen_range(-1.0..1.0)])
        .collect();

    let v_2: Vec<Vec<f64>> = v_1.iter().map(|v1| vec![v1[0] + v1[1]]).collect();

    let dist_cov = dist_cov_multi_exp(&v_1, &v_2);
    let dist_cov_v1 = dist_cov_multi_exp(&v_1, &v_1);
    let dist_cov_v2 = dist_cov_multi_exp(&v_2, &v_2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = dist_cov_multi(&v_1, &v_2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = dist_cov_multi(&v_1, &v_1);
    let dist_cov_stand_v2 = dist_cov_multi(&v_2, &v_2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
fn independent_medium() {
    /*
        let sample_size = 10000;
        let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
        let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

        let v_1: Vec<Vec<f64>> = (0..sample_size)
            .map(move |_x| vec![rng_1.gen_range(-10.0..10.0)])
            .collect();

        let v_2: Vec<Vec<f64>> = (0..sample_size)
            .map(move |_x| vec![rng_2.gen_range(-10.0..10.0)])
            .collect();
    */

    let sample_size = 5000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);
    let mut rng_3 = ChaCha8Rng::seed_from_u64(13);

    let v_1: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_1.gen_range(-1.0..1.0), rng_2.gen_range(-1.0..1.0)])
        .collect();

    /*
    let v_2: Vec<Vec<f64>> = v_1
        .iter()
        .map(|v_i| vec![v_i.iter().sum::<f64>()])
        .collect();
    */
    let v_2: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_3.gen_range(-1.0..1.0)])
        .collect();

    let dist_cov = dist_cov_multi_exp(&v_1, &v_2);
    let dist_cov_v1 = dist_cov_multi_exp(&v_1, &v_1);
    let dist_cov_v2 = dist_cov_multi_exp(&v_2, &v_2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = dist_cov_multi(&v_1, &v_2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = dist_cov_multi(&v_1, &v_1);
    let dist_cov_stand_v2 = dist_cov_multi(&v_2, &v_2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}

#[test]
fn independent_medium_2() {
    let sample_size = 10000;
    let mut rng_1 = ChaCha8Rng::seed_from_u64(134);
    let mut rng_2 = ChaCha8Rng::seed_from_u64(11);

    let v_1: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_1.gen_range(-1.0..1.0)])
        .collect();

    let v_2: Vec<Vec<f64>> = (0..sample_size)
        .map(move |_x| vec![rng_2.gen_range(-1.0..1.0)])
        .collect();

    let dist_cov = dist_cov_multi_exp(&v_1, &v_2);
    let dist_cov_v1 = dist_cov_multi_exp(&v_1, &v_1);
    let dist_cov_v2 = dist_cov_multi_exp(&v_2, &v_2);

    let dist_corr = (dist_cov / (dist_cov_v1 * dist_cov_v2).sqrt()).sqrt();

    println!("Dist corr exp: {:?}", dist_corr);

    let dist_cov_stand = dist_cov_multi(&v_1, &v_2);
    println!("dist cov stand: {:?}", dist_cov_stand);
    let dist_cov_stand_v1 = dist_cov_multi(&v_1, &v_1);
    let dist_cov_stand_v2 = dist_cov_multi(&v_2, &v_2);

    let dist_corr = (dist_cov_stand / (dist_cov_stand_v1 * dist_cov_stand_v2).sqrt()).sqrt();

    println!("Dist corr standard: {:?}", dist_corr);
}
