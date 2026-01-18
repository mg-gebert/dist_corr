// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::api::{DistCorrelation, DistCovariance};

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Tests

/// check if consecutive calculations of
/// distance correlation and distance covariance give exactly the same result
#[test]
fn sin_determinism() {
    let test_sizes = [2_i32.pow(6), 2_i32.pow(11), 121, 597];

    for numb in test_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let v1: Vec<f64> = (0..numb)
            .map(move |_x| rng.random_range(-10.0..10.0))
            .collect();

        let v2: Vec<f64> = v1.iter().map(|x| x.sin()).collect();

        let dist_correlation = DistCorrelation;
        let dist_covariance = DistCovariance;

        let mut dist_corr_prev = dist_correlation.compute(&v1, &v2).unwrap();
        let mut dist_cov_prev = dist_covariance.compute(&v1, &v2).unwrap();
        for _i in 0..100 {
            let dist_corr = dist_correlation.compute(&v1, &v2).unwrap();
            let dist_cov = dist_covariance.compute(&v1, &v2).unwrap();

            assert!((dist_corr - dist_corr_prev).abs() < f64::EPSILON);
            assert!((dist_cov - dist_cov_prev).abs() < f64::EPSILON);

            dist_corr_prev = dist_corr;
            dist_cov_prev = dist_cov;
        }
    }
}
