// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use itertools::izip;
use log::debug;
use rayon::prelude::*;
use std::error::Error;

use crate::frob_inner_product::compute_frobenius_inner_product;
use crate::grand_mean::GrandMeans;
use crate::ordering::Ordering;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

/// computes distance correlation of vectors v1 and v2
pub(crate) fn dist_corr(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let len = v1.len();

    // sort v1,v2 with respect to ordering of v2
    let Ordering {
        v1_per,
        v2_ord,
        order_v1_per,
    } = Ordering::order_wrt_v2(v1, v2, true);

    // compute grand means of v1 and v2
    let grand_means_v1 = GrandMeans::new(&v1_per).compute_unordered(order_v1_per.as_ref().unwrap());
    let grand_means_v2 = GrandMeans::new(&v2_ord).compute_ordered();

    // compute distance variance of v1 and v2
    let dist_var_v1 = dist_var_sq_helper(&v1_per, &grand_means_v1, len as f64).sqrt();
    let dist_var_v2 = dist_var_sq_helper(&v2_ord, &grand_means_v2, len as f64).sqrt();

    if dist_var_v1 > 0.0 && dist_var_v2 > 0.0 {
        // compute distance covariance
        let dist_cov_v1_v2 =
            dist_cov_sq_helper(&v1_per, &v2_ord, &grand_means_v1, &grand_means_v2, len).sqrt();

        Ok(dist_cov_v1_v2 / (dist_var_v1 * dist_var_v2).sqrt())
    } else {
        Ok(0.0)
    }
}

/// computes distance covariance of vectors v1 and v2
pub(crate) fn dist_cov(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let len = v1.len();

    // sort v1,v2 with respect to ordering of v2
    let Ordering {
        v1_per,
        v2_ord,
        order_v1_per,
    } = Ordering::order_wrt_v2(v1, v2, true);

    // compute grand means of v1 and v2
    let grand_means_v1 = GrandMeans::new(&v1_per).compute_unordered(order_v1_per.as_ref().unwrap());
    let grand_means_v2 = GrandMeans::new(&v2_ord).compute_ordered();

    Ok(dist_cov_sq_helper(&v1_per, &v2_ord, &grand_means_v1, &grand_means_v2, len).sqrt())
}

/// computes dVar(v)
pub(crate) fn dist_var(v: &[f64]) -> f64 {
    let len = v.len();

    // sort v
    let mut v_ord = v.to_vec();
    v_ord.par_sort_unstable_by(|v_i, v_j| v_i.partial_cmp(v_j).unwrap());

    // compute grand means
    let grand_means_v = GrandMeans::new(&v_ord).compute_ordered();

    dist_var_sq_helper(v, &grand_means_v, len as f64).sqrt()
}

/// computes dCov^2 from intermediate input
fn dist_cov_sq_helper(
    v1: &[f64],
    v2: &[f64],
    grand_mean_v1: &[f64],
    grand_mean_v2: &[f64],
    len: usize,
) -> f64 {
    // frobenius inner product of distance matrices corresponding to v1 and v2
    let frob_prod_dist_mat = compute_frobenius_inner_product(v1, v2, len);

    // dot product of the grand means of the distance matrices corresponding to v1 and v2
    let dot_prod_grand_means = izip!(grand_mean_v1, grand_mean_v2)
        .map(|(a, b)| a * b)
        .sum::<f64>();

    // 1-norm of distance matrix corresponding to v1
    let dist_mat_v1_one_norm = grand_mean_v1.iter().sum::<f64>();

    // 1-norm of distance matrix corresponding to v2
    let dist_mat_v2_one_norm = grand_mean_v2.iter().sum::<f64>();

    let len_sq = (len * len) as f64;

    let dist_cov_sq = frob_prod_dist_mat / len_sq - 2.0 * dot_prod_grand_means / len as f64
        + dist_mat_v1_one_norm * dist_mat_v2_one_norm / len_sq;

    // dist_cov_sq must be >= 0.0
    // if not there must be a numerical error
    if dist_cov_sq < 0.0 {
        debug!(
            "dist_cov_sq_helper method gives negative: {:?} - use 0.0",
            dist_cov_sq
        );
        0.0
    } else {
        dist_cov_sq
    }
}

/// computes dVar^2 from intermediate input
pub(crate) fn dist_var_sq_helper(v: &[f64], grand_means: &[f64], len: f64) -> f64 {
    let (sum, sum_of_sq) = v.iter().fold((0.0, 0.0), |(sum, sum_of_sq), &x| {
        (sum + x, sum_of_sq + x * x)
    });

    let dist_scalar_prod = 2.0 * len * sum_of_sq - 2.0 * sum.powi(2);
    let len_sq = len * len;

    let dist_var_sq = dist_scalar_prod / len_sq
        - 2.0 * grand_means.iter().map(|a| a * a).sum::<f64>() / len
        + grand_means.iter().sum::<f64>().powi(2) / len_sq;

    if dist_var_sq < 0.0 {
        debug!(
            "dist_var_sq_helper method gives something negative: {:?} - use 0.0",
            dist_var_sq
        );
        0.0
    } else {
        dist_var_sq
    }
}
