// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rayon::prelude::*;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

/// naive implementation of distance covariance with n^2 complexity
///
/// used to test if faster algorithms give correct results
pub fn _dist_cov_sq_naive(data_1: &[f64], data_2: &[f64]) -> f64 {
    let data_length = data_1.len() as f64;
    let dist_hs_norm = data_1
        .par_iter()
        .zip(data_2.par_iter())
        .enumerate()
        .map(|(i, (a1, b1))| {
            data_1[i..]
                .iter()
                .zip(data_2[i..].iter())
                .map(|(a2, b2)| (a1 - a2).abs() * (b1 - b2).abs())
                .sum::<f64>()
        })
        .sum::<f64>()
        * 2.0
        / (data_length * data_length);

    let dist_scalar_avg = data_1
        .par_iter()
        .zip(data_2.par_iter())
        .map(|(a1, b1)| {
            let data1i = data_1.iter().map(|a2| (a1 - a2).abs()).sum::<f64>();
            let data2i = data_2.iter().map(|b2| (b1 - b2).abs()).sum::<f64>();
            data1i * data2i
        })
        .sum::<f64>()
        / (data_length * data_length * data_length);

    let mut mean_a = 0.0;
    let mut mean_b = 0.0;
    data_1.iter().zip(data_2.iter()).for_each(|(a1, b1)| {
        mean_a += data_1.iter().map(|a2| (a1 - a2).abs()).sum::<f64>();
        mean_b += data_2.iter().map(|b2| (b1 - b2).abs()).sum::<f64>();
    });

    let means = mean_a * mean_b / (data_length * data_length * data_length * data_length);

    dist_hs_norm - 2.0 * dist_scalar_avg + means
}

pub fn _dist_cov_naive_exp(v1: &[f64], v2: &[f64]) -> f64 {
    let data_length = v1.len() as f64;
    let dist_frob_norm = (v1
        .par_iter()
        .zip(v2.par_iter())
        .enumerate()
        .map(|(i, (a1, b1))| {
            v1[i..]
                .iter()
                .zip(v2[i..].iter())
                .map(|(a2, b2)| (-(a1 - a2).abs()).exp() * (-(b1 - b2).abs()).exp())
                .sum::<f64>()
        })
        .sum::<f64>()
        * 2.0
        - data_length)
        / (data_length * data_length);

    let dist_scalar_avg = v1
        .par_iter()
        .zip(v2.par_iter())
        .map(|(a1, b1)| {
            let data1i = v1.iter().map(|a2| (-(a1 - a2).abs()).exp()).sum::<f64>();
            let data2i = v2.iter().map(|b2| (-(b1 - b2).abs()).exp()).sum::<f64>();
            data1i * data2i
        })
        .sum::<f64>()
        / (data_length * data_length * data_length);

    let mut mean_a = 0.0;
    let mut mean_b = 0.0;
    v1.iter().zip(v2.iter()).for_each(|(a1, b1)| {
        mean_a += v1.iter().map(|a2| (-(a1 - a2).abs()).exp()).sum::<f64>();
        mean_b += v2.iter().map(|b2| (-(b1 - b2).abs()).exp()).sum::<f64>();
    });

    let means = mean_a * mean_b / (data_length * data_length * data_length * data_length);

    dist_frob_norm - 2.0 * dist_scalar_avg + means
}
