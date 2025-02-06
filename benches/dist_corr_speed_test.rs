// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

use dist_corr::dist_corr;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Bench

fn dist_corr_quadratic_relation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dist_corr_quadratic_relation");

    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let v_1: Vec<f64> = (0..2_i32.pow(16) as usize)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();
    let v_2: Vec<f64> = v_1.iter().map(|x| x * x).collect();
    println!("Test: {:?}", dist_corr(&v_1, &v_2));

    group.bench_function("quadratic_relation", |b| {
        b.iter(|| dist_corr(&v_1, &v_2));
    });

    group.finish();
}

criterion_group!(
    name = dist_corr_speed_test;

    config = Criterion::default()
        //.warm_up_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(60))
        .sample_size(10);

    targets =
        dist_corr_quadratic_relation,

);

criterion_main!(dist_corr_speed_test);
