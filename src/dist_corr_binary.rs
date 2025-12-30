// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rayon::join;
use rayon::prelude::*;
use std::error::Error;

use crate::dist_corr::dist_var_helper;
use crate::grand_mean::{grand_means, grand_means_weighted};

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

// v1 and v2 must be 0-1-valued
pub fn dist_corr_both_binary(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let (n00, n01, n10, n11) = v1.iter().zip(v2.iter()).fold(
        (0.0, 0.0, 0.0, 0.0),
        |(n00, n01, n10, n11), (&a, &b)| match (a, b) {
            (0.0, 0.0) => (n00 + 1.0, n01, n10, n11),
            (0.0, _) => (n00, n01 + 1.0, n10, n11),
            (_, 0.0) => (n00, n01, n10 + 1.0, n11),
            (_, _) => (n00, n01, n10, n11 + 1.0),
        },
    );

    let numerator: f64 = n11 * n00 - n10 * n01;
    let denominator: f64 = ((n11 + n10) * (n11 + n01) * (n00 + n01) * (n00 + n10)).sqrt();

    Ok((numerator / denominator).abs())
}

/// v1 and v2 must be 0-1-valued
pub fn dist_corr_one_binary(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let length = v1.len();

    // sort v1,v2 with respect to ordering of v2
    let (mut v1_transformed, v2_sorted) = order_wrt_v2_simple(v1, v2);

    // initialize grand means
    let mut grand_means_v2 = vec![0.0; length];
    let mut grand_means_v2_weighted = vec![0.0; length];

    v1_transformed
        .iter_mut()
        .for_each(|vi| *vi = 2.0 * *vi - 1.0);

    let ((), ()) = join(
        || {
            grand_means(&v2_sorted, None, &mut grand_means_v2, length);
        },
        || {
            grand_means_weighted(
                &v2_sorted,
                &v1_transformed,
                &mut grand_means_v2_weighted,
                length,
            );
        },
    );

    let dist_var_v2 = dist_var_helper(&v2_sorted, &grand_means_v2, length as f64);
    let dist_var_v1 = dist_cov_both_binary(v1, v1)?;

    let (v1_dist_v1, v1_1, v1_dist_1, dist_1) = v1_transformed
        .iter()
        .zip(grand_means_v2_weighted.iter())
        .zip(grand_means_v2.iter())
        .fold((0.0, 0.0, 0.0, 0.0), |acc, ((vi, vwi_weighted), vwi)| {
            (
                acc.0 + vi * vwi_weighted,
                acc.1 + vi,
                acc.2 + vi * vwi,
                acc.3 + vwi,
            )
        });

    Ok(
        ((-0.5 * v1_dist_v1 / (length as f64) + v1_1 * v1_dist_1 / (length.pow(2) as f64)
            - 0.5 * v1_1.powi(2) * dist_1 / (length.pow(3) as f64))
            / (dist_var_v2 * dist_var_v1).sqrt())
        .sqrt(),
    )
}

// v1 and v2 must be a 0-1-valued
pub fn dist_cov_both_binary(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let (n00, n01, n10, n11) = v1.iter().zip(v2.iter()).fold(
        (0.0, 0.0, 0.0, 0.0),
        |(n00, n01, n10, n11), (&a, &b)| match (a, b) {
            (0.0, 0.0) => (n00 + 1.0, n01, n10, n11),
            (0.0, _) => (n00, n01 + 1.0, n10, n11),
            (_, 0.0) => (n00, n01, n10 + 1.0, n11),
            (_, _) => (n00, n01, n10, n11 + 1.0),
        },
    );

    Ok((2.0 * (n11 * n00 - n10 * n01) / (v1.len() as f64).powi(2)).powi(2))
}

/// v1 must be 0-1-valued
pub fn dist_cov_one_binary(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let length = v1.len();

    // sort v1,v2 with respect to ordering of v2
    let (mut v1_transformed, v2_sorted) = order_wrt_v2_simple(v1, v2);

    //let tick = Instant::now();
    // initialize grand means
    //let mut grand_means_v2 = vec![0.0; length];
    let mut grand_means_v2_weighted = vec![0.0; length];

    v1_transformed
        .iter_mut()
        .for_each(|vi| *vi = 2.0 * *vi - 1.0);
    //println!("Initialising took {}s", tick.elapsed().as_secs_f32());

    let v1_transformed_sum = v1_transformed.iter().sum::<f64>() / length as f64;
    v1_transformed
        .iter_mut()
        .for_each(|vi| *vi -= v1_transformed_sum);

    grand_means_weighted(
        &v2_sorted,
        &v1_transformed,
        &mut grand_means_v2_weighted,
        length,
    );

    Ok(-v1_transformed
        .iter()
        .zip(grand_means_v2_weighted)
        .map(|(vi, grand_mean_i)| vi * grand_mean_i)
        .sum::<f64>()
        / (2.0 * length as f64))
}

fn order_wrt_v2_simple(v1: &[f64], v2: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Create a sorted list of indices based on v2
    let mut indices: Vec<usize> = (0..v2.len()).collect();
    indices.par_sort_unstable_by(|&i, &j| v2[i].partial_cmp(&v2[j]).unwrap());

    // Map sorted indices to values in v1 and v2
    indices.iter().map(|&i| (v1[i], v2[i])).unzip()
}
