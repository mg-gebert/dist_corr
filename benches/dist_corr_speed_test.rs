// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use std::time::Duration;

use dist_corr::DistCorrelation;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Bench

fn dist_corr_small(c: &mut Criterion) {
    let thread_counts = [0, 1, 2, 4, 8, 16]; // 0 = sequential baseline
    let (v1, v2) = samples_random(1024, 76, |x| x * x);

    let mut group = c.benchmark_group("Small");

    for &threads in &thread_counts {
        // threads = 0 means parallel = false
        if threads == 0 {
            let dist_corr = DistCorrelation::new(false);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| b.iter(|| dist_corr.compute(&v1, &v2)),
            );
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let dist_corr = DistCorrelation::new(true);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| {
                    pool.install(|| {
                        b.iter(|| dist_corr.compute(&v1, &v2));
                    });
                },
            );
        }
    }
}

fn dist_corr_little(c: &mut Criterion) {
    let thread_counts = [0, 1, 2, 4, 8, 16]; // 0 = sequential baseline
    let (v1, v2) = samples_random(8013, 76, |x| x * x);

    let mut group = c.benchmark_group("Little");

    for &threads in &thread_counts {
        // threads = 0 means parallel = false
        if threads == 0 {
            let dist_corr = DistCorrelation::new(false);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| b.iter(|| dist_corr.compute(&v1, &v2)),
            );
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let dist_corr = DistCorrelation::new(true);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| {
                    pool.install(|| {
                        b.iter(|| dist_corr.compute(&v1, &v2));
                    });
                },
            );
        }
    }
}

fn dist_corr_medium(c: &mut Criterion) {
    let thread_counts = [0, 1, 2, 4, 8, 16]; // 0 = sequential baseline
    let (v1, v2) = samples_random(2_u64.pow(15) as usize, 76, |x| x * x);

    let mut group = c.benchmark_group("Medium");

    for &threads in &thread_counts {
        // threads = 0 means parallel = false
        if threads == 0 {
            let dist_corr = DistCorrelation::new(false);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| b.iter(|| dist_corr.compute(&v1, &v2)),
            );
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let dist_corr = DistCorrelation::new(true);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| {
                    pool.install(|| {
                        b.iter(|| dist_corr.compute(&v1, &v2));
                    });
                },
            );
        }
    }
}

fn dist_corr_big(c: &mut Criterion) {
    let thread_counts = [0, 1, 2, 4, 8, 16]; // 0 = sequential baseline
    let (v1, v2) = samples_random(2_u64.pow(20) as usize, 76, |x| x * x);

    let mut group = c.benchmark_group("Big");

    for &threads in &thread_counts {
        // threads = 0 means parallel = false
        if threads == 0 {
            let dist_corr = DistCorrelation::new(false);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| b.iter(|| dist_corr.compute(&v1, &v2)),
            );
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let dist_corr = DistCorrelation::new(true);
            group.bench_with_input(
                BenchmarkId::new("Threads", threads),
                &threads,
                |b, &_threads| {
                    pool.install(|| {
                        b.iter(|| dist_corr.compute(&v1, &v2));
                    });
                },
            );
        }
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
        dist_corr_little,
        dist_corr_medium,
        dist_corr_big,

);

criterion_main!(dist_corr_speed_test);

fn samples_random(sample_size: usize, seed: u64, func: fn(&f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng.random_range(-10.0..10.0))
        .collect();
    let v2: Vec<f64> = v1.iter().map(func).collect();

    (v1, v2)
}
