// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

use dist_corr::DistCorrelation;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Bench

fn one_binary_standard(c: &mut Criterion) {
    let sample_size_exp: [u32; 9] = [6, 8, 10, 12, 14, 16, 18, 20, 22];

    let mut group = c.benchmark_group("one-binary-standard");

    let dist_corr = DistCorrelation;

    for &exp in &sample_size_exp {
        let (v1, v2) =
            samples_random_by_func(2_usize.pow(exp), 76, |x| if *x > 0.0 { 1.0 } else { 0.0 });
        println!(
            "Exp: {:?} - dist corr: {:?}",
            exp,
            dist_corr.compute(&v1, &v2)
        );

        group.bench_with_input(BenchmarkId::new("Exponent", exp), &exp, |b, &_exp| {
            b.iter(|| dist_corr.compute(&v1, &v2));
        });
    }
}

fn one_binary_semi_binary(c: &mut Criterion) {
    let sample_size_exp: [u32; 9] = [6, 8, 10, 12, 14, 16, 18, 20, 22];

    let mut group = c.benchmark_group("one-binary-semi-binary");

    let dist_corr = DistCorrelation;

    for &exp in &sample_size_exp {
        let (v1, v2) =
            samples_random_by_func(2_usize.pow(exp), 76, |x| if *x > 0.0 { 1.0 } else { 0.0 });
        println!(
            "Exp: {:?} - dist corr: {:?}",
            exp,
            dist_corr.compute_binary(&v1, &v2, false, true)
        );

        group.bench_with_input(BenchmarkId::new("Exponent", exp), &exp, |b, &_exp| {
            b.iter(|| dist_corr.compute_binary(&v1, &v2, false, true));
        });
    }
}

criterion_group!(
    name = dist_corr_one_binary;

    config = Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(100))
        .sample_size(10);

    targets =
        one_binary_standard,
        one_binary_semi_binary,

);

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Bench

fn both_binary_standard(c: &mut Criterion) {
    let sample_size_exp: [u32; _] = [6, 8, 10, 12, 14, 16, 18, 20, 22];

    let mut group = c.benchmark_group("both-binary-standard");

    let dist_corr = DistCorrelation;

    for &exp in &sample_size_exp {
        let (v1, v2) = samples_random_two_binary(2_usize.pow(exp), 76);
        println!(
            "Exp: {:?} - dist corr: {:?}",
            exp,
            dist_corr.compute(&v1, &v2)
        );

        group.bench_with_input(BenchmarkId::new("Exponent", exp), &exp, |b, &_exp| {
            b.iter(|| dist_corr.compute(&v1, &v2));
        });
    }
}

fn both_binary_semi_binary(c: &mut Criterion) {
    let sample_size_exp: [u32; _] = [6, 8, 10, 12, 14, 16, 18, 20, 22];

    let mut group = c.benchmark_group("both-binary-semi-binary");

    let dist_corr = DistCorrelation;

    for &exp in &sample_size_exp {
        let (v1, v2) = samples_random_two_binary(2_usize.pow(exp), 76);
        println!(
            "Exp: {:?} - dist corr: {:?}",
            exp,
            dist_corr.compute(&v1, &v2)
        );

        group.bench_with_input(BenchmarkId::new("Exponent", exp), &exp, |b, &_exp| {
            b.iter(|| dist_corr.compute_binary(&v1, &v2, false, true));
        });
    }
}

fn both_binary_full_binary(c: &mut Criterion) {
    let sample_size_exp: [u32; _] = [6, 8, 10, 12, 14, 16, 18, 20, 22];

    let mut group = c.benchmark_group("both-binary-full-binary");

    let dist_corr = DistCorrelation;

    for &exp in &sample_size_exp {
        let (v1, v2) = samples_random_two_binary(2_usize.pow(exp), 76);
        println!(
            "Exp: {:?} - dist corr: {:?}",
            exp,
            dist_corr.compute(&v1, &v2)
        );

        group.bench_with_input(BenchmarkId::new("Exponent", exp), &exp, |b, &_exp| {
            b.iter(|| dist_corr.compute_binary(&v1, &v2, true, true));
        });
    }
}

criterion_group!(
    name = dist_corr_both_binary;

    config = Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(100))
        .sample_size(10);

    targets =
        both_binary_standard,
        both_binary_semi_binary,
        both_binary_full_binary

);

criterion_main!(dist_corr_one_binary, dist_corr_both_binary);

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Helper

fn samples_random_two_binary(len: usize, seed_1: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng_1 = ChaCha8Rng::seed_from_u64(seed_1);

    let v1: Vec<f64> = (0..len)
        .map(move |_x| rng_1.gen_range(-10.0..10.0))
        .map(|x| if x < 0.0 { 0.0 } else { 1.0 })
        .collect();

    let v2: Vec<f64> = v1
        .iter()
        .enumerate()
        .map(|(i, x)| if i % 2 == 0 { *x } else { 0.0 })
        .collect();

    (v1, v2)
}

fn samples_random_by_func(len: usize, seed: u64, func: fn(&f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v1: Vec<f64> = (0..len).map(move |_x| rng.gen_range(-10.0..10.0)).collect();
    let v2: Vec<f64> = v1.iter().map(func).collect();

    (v1, v2)
}
