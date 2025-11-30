// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rayon::join;
use rayon::prelude::*;

use std::error::Error;

use crate::frob_inner_product::compute_frobenius_inner_product;
use crate::grand_mean::grand_means;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

pub fn dist_corr_fast(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let v1_len = v1.len();

    // initialize ordering
    let mut ordering: Vec<usize> = (0..v1_len).collect();
    // sort v1,v2 with respect to ordering of v2
    let (v1_shuffled, v2_sorted) = order_wrt_v2(v1, v2, &mut ordering);

    // initialize grand means
    let mut grand_means_v1 = vec![0.0; v1_len];
    let mut grand_means_v2 = vec![0.0; v1_len];

    // compute distance variance of v1 and v2
    // update grand means with grand means of v1 and v2
    let (dist_var_v1, dist_var_v2) = join(
        || {
            grand_means(&v1_shuffled, Some(&ordering), &mut grand_means_v1, v1_len);
            dist_var_fast_helper(&v1_shuffled, &grand_means_v1, v1_len as f64)
        },
        || {
            grand_means(&v2_sorted, None, &mut grand_means_v2, v1_len);
            dist_var_fast_helper(&v2_sorted, &grand_means_v2, v1_len as f64)
        },
    );

    let dist_cov_v1_v2 = dist_cov_fast_helper(
        &v1_shuffled,
        &v2_sorted,
        &grand_means_v1,
        &grand_means_v2,
        v1_len,
    );

    Ok((dist_cov_v1_v2 / (dist_var_v1 * dist_var_v2).sqrt()).sqrt())
}

pub fn dist_cov_fast(v1: &[f64], v2: &[f64]) -> Result<f64, Box<dyn Error>> {
    let v1_len = v1.len();

    let mut ordering: Vec<usize> = (0..v1_len).collect();
    // sort v1,v2 with respect to ordering of v2
    let (v1_shuffled, v2_sorted) = order_wrt_v2(v1, v2, &mut ordering);

    // initialize grand means
    let mut grand_means_v1 = vec![0.0; v1_len];
    let mut grand_means_v2 = vec![0.0; v1_len];

    // update grand means with grand means of v1 and v2
    let ((), ()) = join(
        || {
            grand_means(&v1_shuffled, Some(&ordering), &mut grand_means_v1, v1_len);
        },
        || {
            grand_means(&v2_sorted, None, &mut grand_means_v2, v1_len);
        },
    );

    Ok(dist_cov_fast_helper(
        &v1_shuffled,
        &v2_sorted,
        &grand_means_v1,
        &grand_means_v2,
        v1_len,
    ))
}

pub fn dist_var_fast(v: &[f64]) -> f64 {
    let v_len = v.len();

    assert!(v_len > 0, "v must not be empty.");

    // sort v
    let mut v_sorted = v.to_vec();
    v_sorted.par_sort_unstable_by(|v_i, v_j| v_i.partial_cmp(v_j).unwrap());

    // initialize grand means
    let mut grand_means_v_sorted = vec![0.0; v_len];
    // update grand means
    grand_means(&v_sorted, None, &mut grand_means_v_sorted, v_len);

    dist_var_fast_helper(v, &grand_means_v_sorted, v_len as f64)
}

pub fn dist_cov_fast_helper(
    v1: &[f64],
    v2: &[f64],
    grand_mean_v1: &[f64],
    grand_mean_v2: &[f64],
    len: usize,
) -> f64 {
    let prod = compute_frobenius_inner_product(v1, v2, len);

    prod / (len * len) as f64
        - 2.0
            * grand_mean_v1
                .iter()
                .zip(grand_mean_v2.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
            / (len) as f64
        + grand_mean_v1.iter().sum::<f64>() * grand_mean_v2.iter().sum::<f64>() / (len * len) as f64
}

pub fn dist_var_fast_helper(v: &[f64], grand_means: &[f64], len: f64) -> f64 {
    let (sum, sum_of_sq) = v.iter().fold((0.0, 0.0), |(sum, sum_of_sq), &x| {
        (sum + x, sum_of_sq + x * x)
    });

    let dist_scalar_prod = 2.0 * len * sum_of_sq - 2.0 * sum.powi(2);
    let len_sq = len * len;

    dist_scalar_prod / len_sq - 2.0 * grand_means.iter().map(|a| a * a).sum::<f64>() / len
        + grand_means.iter().sum::<f64>().powi(2) / len_sq
}

/// return
/// (Vec<f64>, Vec<f64>: (v1_shuffled, v2_ordered)
/// where (v1_shuffled, v2_ordered)
/// are a simultanously permutation of (v1, v2)
/// such that v2 is sorted increasingly
///
/// ordering is adapted to reflect the indices changes needed to sort v1
fn order_wrt_v2(v1: &[f64], v2: &[f64], ordering: &mut [usize]) -> (Vec<f64>, Vec<f64>) {
    // compute ordering of v2
    ordering.par_sort_unstable_by(|&i, &j| v2[i].partial_cmp(&v2[j]).unwrap());

    // sort v1 and v2 according to above ordering of v2
    let (v1_shuffled, v2_ordered): (Vec<f64>, Vec<f64>) =
        ordering.iter().map(|&i| (v1[i], v2[i])).unzip();

    // update ordering to reflect ordering of v1
    ordering.par_sort_unstable_by(|&i, &j| v1_shuffled[i].partial_cmp(&v1_shuffled[j]).unwrap());

    (v1_shuffled, v2_ordered)
}

pub fn order_wrt_v2_simple(v1: &[f64], v2: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Create a sorted list of indices based on v2
    let mut indices: Vec<usize> = (0..v2.len()).collect();
    indices.par_sort_unstable_by(|&i, &j| v2[i].partial_cmp(&v2[j]).unwrap());

    // Map sorted indices to values in v1 and v2
    indices.iter().map(|&i| (v1[i], v2[i])).unzip()
}
