// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use dist_corr::DistCorrelation;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Bench

fn dist_corr_small(c: &mut Criterion) {
    let (v1, v2) = samples(1024);

    let mut group = c.benchmark_group("Small");

    let dist_corr = DistCorrelation;

    println!(
        "n: {:} - dist_corr: {:?}",
        1024,
        dist_corr.compute(&v1, &v2)
    );

    group.bench_function("Small", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

fn dist_corr_little(c: &mut Criterion) {
    let (v1, v2) = samples(8013);

    let mut group = c.benchmark_group("Little");

    let dist_corr = DistCorrelation;

    println!(
        "n: {:} - dist_corr: {:?}",
        8013,
        dist_corr.compute(&v1, &v2)
    );

    group.bench_function("Little", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

fn dist_corr_medium(c: &mut Criterion) {
    let (v1, v2) = samples(2_u64.pow(15) as usize);

    let mut group = c.benchmark_group("Medium");

    let dist_corr = DistCorrelation;

    println!(
        "n: {:} - dist_corr: {:?}",
        2_u64.pow(15),
        dist_corr.compute(&v1, &v2)
    );

    group.bench_function("Medium", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

fn dist_corr_big(c: &mut Criterion) {
    let (v1, v2) = samples(2_u64.pow(20) as usize);

    let mut group = c.benchmark_group("Big");

    let dist_corr = DistCorrelation;

    println!(
        "n: {:} - dist_corr: {:?}",
        2_u64.pow(20),
        dist_corr.compute(&v1, &v2)
    );

    group.bench_function("Big", |b| {
        b.iter(|| dist_corr.compute(&v1, &v2));
    });
}

criterion_group!(
    name = dist_corr_speed_comparison;

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

criterion_main!(dist_corr_speed_comparison);

fn samples(sample_size: usize) -> (Vec<f64>, Vec<f64>) {
    let v1: Vec<f64> = (0..sample_size).map(|i| (i as f64).sin()).collect();
    let v2: Vec<f64> = (0..sample_size).map(|i| (i as f64).cos()).collect();

    (v1, v2)
}
