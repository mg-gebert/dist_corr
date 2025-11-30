// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::ParallelSliceMut;
use rayon::ThreadPoolBuilder;
use std::time::Duration;

use dist_corr::frob_inner_product::compute_frobenius_inner_product;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Benchcargo

fn frob_inner_small(c: &mut Criterion) {
    let thread_counts = [1, 2, 4, 8, 16]; // Define different thread counts
    let (v_1, v_2) = samples_random(1024, 76, |x| x * x);
    let len = v_1.len();
    let mut ordering: Vec<usize> = (0..len).collect();
    // sort v_1,v_2 with respect to ordering of v_2
    let (v1_shuffled, v2_sorted) = order_wrt_v2(&v_1, &v_2, &mut ordering);

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
                    b.iter(|| compute_frobenius_inner_product(&v1_shuffled, &v2_sorted, len));
                });
            },
        );
    }
}

fn frob_inner_medium(c: &mut Criterion) {
    let thread_counts = [1, 2, 4, 8, 16]; // Define different thread counts
    let (v_1, v_2) = samples_random(2_u64.pow(15) as usize, 76, |x| x * x);
    let len = v_1.len();
    let mut ordering: Vec<usize> = (0..len).collect();
    // sort v_1,v_2 with respect to ordering of v_2
    let (v1_shuffled, v2_sorted) = order_wrt_v2(&v_1, &v_2, &mut ordering);

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
                    b.iter(|| compute_frobenius_inner_product(&v1_shuffled, &v2_sorted, len));
                });
            },
        );
    }
}

fn frob_inner_big(c: &mut Criterion) {
    let thread_counts = [1, 2, 4, 8, 16]; // Define different thread counts
    let (v_1, v_2) = samples_random(2_u64.pow(20) as usize, 76, |x| x * x);
    let len = v_1.len();
    let mut ordering: Vec<usize> = (0..len).collect();
    // sort v_1,v_2 with respect to ordering of v_2
    let (v1_shuffled, v2_sorted) = order_wrt_v2(&v_1, &v_2, &mut ordering);

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
                    b.iter(|| compute_frobenius_inner_product(&v1_shuffled, &v2_sorted, len));
                });
            },
        );
    }
}

criterion_group!(
    name = frob_inner_prod_speed_test;

    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(60))
        .sample_size(10);

    targets =
        frob_inner_small,
        frob_inner_medium,
        frob_inner_big,

);

criterion_main!(frob_inner_prod_speed_test);

fn samples_random(sample_size: usize, seed: u64, func: fn(&f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let v_1: Vec<f64> = (0..sample_size)
        .map(move |_x| rng.gen_range(-10.0..10.0))
        .collect();
    let v_2: Vec<f64> = v_1.iter().map(func).collect();

    (v_1, v_2)
}

fn order_wrt_v2(v_1: &[f64], v_2: &[f64], ordering: &mut [usize]) -> (Vec<f64>, Vec<f64>) {
    // compute ordering of v_2
    ordering.par_sort_unstable_by(|&i, &j| v_2[i].partial_cmp(&v_2[j]).unwrap());

    // sort v_1 and v_2 according to above ordering of v_2
    let (v1_shuffled, v2_ordered): (Vec<f64>, Vec<f64>) =
        ordering.iter().map(|&i| (v_1[i], v_2[i])).unzip();

    // update ordering to reflect ordering of v_1
    ordering.par_sort_unstable_by(|&i, &j| v1_shuffled[i].partial_cmp(&v1_shuffled[j]).unwrap());

    (v1_shuffled, v2_ordered)
}
