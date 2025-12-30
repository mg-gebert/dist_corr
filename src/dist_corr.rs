// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rayon::join;
use rayon::prelude::*;

use std::error::Error;

use crate::frob_inner_product::compute_frobenius_inner_product;
use crate::grand_mean::grand_means;
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

    // initialize grand means
    let mut grand_means_v1 = vec![0.0; len];
    let mut grand_means_v2 = vec![0.0; len];

    // compute distance variance of v1 and v2
    // update grand means with grand means of v1 and v2
    let (dist_var_v1, dist_var_v2) = join(
        || {
            grand_means(&v1_per, order_v1_per.as_deref(), &mut grand_means_v1, len);
            dist_var_helper(&v1_per, &grand_means_v1, len as f64)
        },
        || {
            grand_means(&v2_ord, None, &mut grand_means_v2, len);
            dist_var_helper(&v2_ord, &grand_means_v2, len as f64)
        },
    );

    // compute distance covariance
    let dist_cov_v1_v2 = dist_cov_helper(&v1_per, &v2_ord, &grand_means_v1, &grand_means_v2, len);

    Ok((dist_cov_v1_v2 / (dist_var_v1 * dist_var_v2).sqrt()).sqrt())
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

    // initialize grand means
    let mut grand_means_v1 = vec![0.0; len];
    let mut grand_means_v2 = vec![0.0; len];

    // update grand means with grand means of v1 and v2
    let ((), ()) = join(
        || {
            grand_means(&v1_per, order_v1_per.as_deref(), &mut grand_means_v1, len);
        },
        || {
            grand_means(&v2_ord, None, &mut grand_means_v2, len);
        },
    );

    Ok(dist_cov_helper(
        &v1_per,
        &v2_ord,
        &grand_means_v1,
        &grand_means_v2,
        len,
    ))
}

/// computes distance variance of vector v
pub(crate) fn dist_var(v: &[f64]) -> f64 {
    let len = v.len();

    // sort v
    let mut v_sorted = v.to_vec();
    v_sorted.par_sort_unstable_by(|v_i, v_j| v_i.partial_cmp(v_j).unwrap());

    // initialize grand means
    let mut grand_means_v_sorted = vec![0.0; len];
    // update grand means
    grand_means(&v_sorted, None, &mut grand_means_v_sorted, len);

    dist_var_helper(v, &grand_means_v_sorted, len as f64)
}

fn dist_cov_helper(
    v1: &[f64],
    v2: &[f64],
    grand_mean_v1: &[f64],
    grand_mean_v2: &[f64],
    len: usize,
) -> f64 {
    // frobenius inner product of distance matrices corresponding to v1 and v2
    let frob_prod_dist_mat = compute_frobenius_inner_product(v1, v2, len);

    // dot product of the grand means of the distance matrices corresponding to v1 and v2
    let dot_prod_grand_means = grand_mean_v1
        .iter()
        .zip(grand_mean_v2.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();

    // 1-norm of distance matrix corresponding to v1
    let dist_mat_v1_one_norm = grand_mean_v1.iter().sum::<f64>();

    // 1-norm of distance matrix corresponding to v2
    let dist_mat_v2_one_norm = grand_mean_v2.iter().sum::<f64>();

    let len_sq = (len * len) as f64;

    frob_prod_dist_mat / len_sq - 2.0 * dot_prod_grand_means / len as f64
        + dist_mat_v1_one_norm * dist_mat_v2_one_norm / len_sq
}

pub(crate) fn dist_var_helper(v: &[f64], grand_means: &[f64], len: f64) -> f64 {
    let (sum, sum_of_sq) = v.iter().fold((0.0, 0.0), |(sum, sum_of_sq), &x| {
        (sum + x, sum_of_sq + x * x)
    });

    let dist_scalar_prod = 2.0 * len * sum_of_sq - 2.0 * sum.powi(2);
    let len_sq = len * len;

    dist_scalar_prod / len_sq - 2.0 * grand_means.iter().map(|a| a * a).sum::<f64>() / len
        + grand_means.iter().sum::<f64>().powi(2) / len_sq
}
