// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

use dist_corr::DistCorrelation;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Bench

fn dist_corr_small(c: &mut Criterion) {
    let (v1, v2) = samples_random(1024, 76, |x| x * x);

    let mut group = c.benchmark_group("Small");

    let dist_corr = DistCorrelation;

    group.bench_function("Small", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

fn dist_corr_little(c: &mut Criterion) {
    let (v1, v2) = samples_random(8013, 76, |x| x * x);

    let mut group = c.benchmark_group("Little");

    let dist_corr = DistCorrelation;

    group.bench_function("Little", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

fn dist_corr_medium(c: &mut Criterion) {
    let (v1, v2) = samples_random(2_u64.pow(15) as usize, 76, |x| x * x);

    let mut group = c.benchmark_group("Medium");

    let dist_corr = DistCorrelation;

    group.bench_function("Medium", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

fn dist_corr_big(c: &mut Criterion) {
    let (v1, v2) = samples_random(2_u64.pow(20) as usize, 76, |x| x * x);

    let mut group = c.benchmark_group("Big");

    let dist_corr = DistCorrelation;

    group.bench_function("Big", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

criterion_group!(
    name = dist_corr_speed_default_test;

    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(60))
        .sample_size(10);

    targets =
        dist_corr_small,
        dist_corr_little,
        dist_corr_medium,
        dist_corr_big,

);

criterion_main!(dist_corr_speed_default_test);

fn samples_random(sample_size: usize, seed: u64, func: fn(&f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();
    let v2: Vec<f64> = v1.iter().map(func).collect();

    (v1, v2)
}
