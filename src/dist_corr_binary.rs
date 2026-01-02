// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rayon::join;
use std::error::Error;

use crate::dist_corr::dist_var_helper;
use crate::grand_mean::GrandMeans;
use crate::ordering::Ordering;

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

/// v1 must be 0-1-valued
pub fn dist_corr_one_binary(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let length = v1.len();

    // sort v1,v2 with respect to ordering of v2
    let Ordering {
        mut v1_per, v2_ord, ..
    } = Ordering::order_wrt_v2(v1, v2, false);

    // initialize grand means
    let mut grand_means_v2 = GrandMeans::new(&v2_ord);
    let mut grand_means_v2_weighted = GrandMeans::new(&v2_ord);

    v1_per.iter_mut().for_each(|vi| *vi = 2.0 * *vi - 1.0);

    let ((), ()) = join(
        || {
            grand_means_v2.compute_ordered();
        },
        || {
            grand_means_v2_weighted.compute_ordered_weighted(&v1_per);
        },
    );

    let dist_var_v2 = dist_var_helper(&v2_ord, grand_means_v2.get_means(), length as f64);
    let dist_var_v1 = dist_cov_both_binary(v1, v1)?;

    let (v1_dist_v1, v1_1, v1_dist_1, dist_1) = v1_per
        .iter()
        .zip(grand_means_v2_weighted.get_means().iter())
        .zip(grand_means_v2.get_means().iter())
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
    let Ordering {
        mut v1_per, v2_ord, ..
    } = Ordering::order_wrt_v2(v1, v2, false);

    let mut grand_means_v2_weighted = GrandMeans::new(&v2_ord);

    v1_per.iter_mut().for_each(|vi| *vi = 2.0 * *vi - 1.0);

    let v1_transformed_sum = v1_per.iter().sum::<f64>() / length as f64;
    v1_per.iter_mut().for_each(|vi| *vi -= v1_transformed_sum);

    grand_means_v2_weighted.compute_ordered_weighted(&v1_per);

    Ok(-v1_per
        .iter()
        .zip(grand_means_v2_weighted.get_means())
        .map(|(vi, grand_mean_i)| vi * grand_mean_i)
        .sum::<f64>()
        / (2.0 * length as f64))
}
