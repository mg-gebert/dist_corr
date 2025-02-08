// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use std::time::Duration;

use dist_corr::dist_corr;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Bench

fn dist_corr_small(c: &mut Criterion) {
    let thread_counts = [1, 2, 4, 8, 16]; // Define different thread counts
    let (v_1, v_2) = samples_random(1024, 76, |x| x * x);

    let mut group = c.benchmark_group("Small");

    for &threads in &thread_counts {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("Threads", threads),
            &threads,
            |b, &_threads| {
                pool.install(|| {
                    b.iter(|| dist_corr(&v_1, &v_2));
                });
            },
        );
    }
}

fn dist_corr_medium(c: &mut Criterion) {
    let thread_counts = [1, 2, 4, 8, 16]; // Define different thread counts
    let (v_1, v_2) = samples_random(2_u64.pow(15) as usize, 76, |x| x * x);

    let mut group = c.benchmark_group("Medium");

    for &threads in &thread_counts {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("Threads", threads),
            &threads,
            |b, &_threads| {
                pool.install(|| {
                    b.iter(|| dist_corr(&v_1, &v_2));
                });
            },
        );
    }
}

fn dist_corr_big(c: &mut Criterion) {
    let thread_counts = [1, 2, 4, 8, 16]; // Define different thread counts
    let (v_1, v_2) = samples_random(2_u64.pow(20) as usize, 76, |x| x * x);

    let mut group = c.benchmark_group("Big");

    for &threads in &thread_counts {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("Threads", threads),
            &threads,
            |b, &_threads| {
                pool.install(|| {
                    b.iter(|| dist_corr(&v_1, &v_2));
                });
            },
        );
    }
}

criterion_group!(
    name = dist_corr_speed_test;

    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(60))
        .sample_size(10);

    targets =
        dist_corr_small,
        dist_corr_medium,
        dist_corr_big,

);

criterion_main!(dist_corr_speed_test);

fn samples_random(sample_size: usize, seed: u64, func: fn(&f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v_1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();
    let v_2: Vec<f64> = v_1.iter().map(func).collect();

    (v_1, v_2)
}
