// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use rayon::prelude::*;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Struct

pub(crate) struct Ordering {
    pub v1_per: Vec<f64>,
    pub v2_ord: Vec<f64>,
    pub order_v1_per: Option<Vec<usize>>,
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

impl Ordering {
    /// Simultaneously permutes two slices so that `v2` becomes sorted, and then returns the permuted vectors and the ordering of the first vector.
    ///
    /// # Arguments
    ///
    /// * `v1` - A slice of floating-point values.
    /// * `v2` - A slice of floating-point values.
    /// * `store_order_v1` - bool.
    ///
    /// # Returns
    ///
    /// Ordering with
    /// - `v1_per`: A `Vec<f64>` with elements of `v1` reordered to match the permutation that sorts `v2`.
    /// - `v2_ord`: A `Vec<f64>` which is a sorted version of `v2` (in increasing order).
    /// - `ordering`: A `Vec<usize>` representing the indices that would sort `v1_per` in increasing order.
    pub(crate) fn order_wrt_v2(v1: &[f64], v2: &[f64], store_order_v1: bool) -> Ordering {
        let mut ordering: Vec<usize> = (0..v1.len()).collect();

        // compute ordering of v2
        ordering.par_sort_unstable_by(|&i, &j| v2[i].partial_cmp(&v2[j]).unwrap());

        // sort v1 and v2 according to above ordering of v2
        let (v1_per, v2_ord): (Vec<f64>, Vec<f64>) =
            ordering.iter().map(|&i| (v1[i], v2[i])).unzip();

        if store_order_v1 {
            // update ordering to reflect ordering of v1_shuffled
            ordering.par_sort_unstable_by(|&i, &j| v1_per[i].partial_cmp(&v1_per[j]).unwrap());

            Ordering {
                v1_per,
                v2_ord,
                order_v1_per: Some(ordering),
            }
        } else {
            Ordering {
                v1_per,
                v2_ord,
                order_v1_per: None,
            }
        }
    }
}
