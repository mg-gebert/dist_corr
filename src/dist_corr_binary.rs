// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use crate::dist_corr_fast::{dist_var_fast_helper, order_wrt_v2_simple};
use crate::grand_mean::{grand_means, grand_means_weighted};
use rayon::join;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

pub fn dist_corr_fast_binary(v_1: &[f64], v_2: &[f64]) -> f64 {
    let (n00, n01, n10, n11) =
        v_1.iter()
            .zip(v_2.iter())
            .fold(
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

    -numerator / denominator
}

/// v_1 should be binary
pub fn dist_corr_fast_one_binary(v_1: &[f64], v_2: &[f64]) -> f64 {
    let length = v_1.len();

    // sort v_1,v_2 with respect to ordering of v_2
    let (mut v1_transformed, v2_sorted) = order_wrt_v2_simple(v_1, v_2);

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

    let dist_var_v2 = dist_var_fast_helper(&v2_sorted, &grand_means_v2, length as f64);
    let dist_var_v1_root = dist_cov_binary_sqrt(v_1, v_1);

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

    ((-0.5 * v1_dist_v1 / (length as f64) + v1_1 * v1_dist_1 / (length.pow(2) as f64)
        - 0.5 * v1_1.powi(2) * dist_1 / (length.pow(3) as f64))
        / (dist_var_v2.sqrt() * dist_var_v1_root))
        .sqrt()
}

pub fn dist_cov_binary_sqrt(v_1: &[f64], v_2: &[f64]) -> f64 {
    let length = v_1.len() as f64;

    let (v1_sum, v2_sum, v1_v2_sum) = v_1.iter().zip(v_2.iter()).fold(
        (0.0, 0.0, 0.0),
        |(v1_sum, v2_sum, v1_v2_sum), (&v1_i, &v2_i)| {
            (
                v1_sum + (2.0 * v1_i - 1.0),
                v2_sum + (2.0 * v2_i - 1.0),
                v1_v2_sum + (2.0 * v1_i - 1.0) * (2.0 * v2_i - 1.0),
            )
        },
    );

    (v1_v2_sum - v1_sum * v2_sum / length).abs() / (2.0 * length)
}

/// v_1 should be binary
pub fn dist_cov_one_binary(v_1: &[f64], v_2: &[f64]) -> f64 {
    let length = v_1.len();

    // sort v_1,v_2 with respect to ordering of v_2
    let (mut v1_transformed, v2_sorted) = order_wrt_v2_simple(v_1, v_2);

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

    -v1_transformed
        .iter()
        .zip(grand_means_v2_weighted)
        .map(|(vi, grand_mean_i)| vi * grand_mean_i)
        .sum::<f64>()
        / (2.0 * length as f64)
}
