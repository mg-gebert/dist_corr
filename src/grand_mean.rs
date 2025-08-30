use itertools::izip;

/// out must be all 0s
pub fn grand_means(v: &[f64], order: Option<&[usize]>, out: &mut [f64], len: usize) {
    let mut current_sum_ascending = 0.0;
    let mut current_sum_descending = 0.0;

    if let Some(order) = order {
        izip!(order.iter(), order.iter().rev())
            .enumerate()
            .for_each(|(i, (ord_j, rev_ord_j))| {
                out[*ord_j] +=
                    (2 * (i as i64) - len as i64 + 1) as f64 * v[*ord_j] - current_sum_ascending;
                out[*rev_ord_j] += current_sum_descending;
                current_sum_descending += v[*rev_ord_j];
                current_sum_ascending += v[*ord_j];
            });
    } else {
        izip!(v.iter(), v.iter().rev())
            .enumerate()
            .for_each(|(i, (v_i, rev_v_i))| {
                out[i] += (2 * (i as i64) - len as i64 + 1) as f64 * v_i - current_sum_ascending;
                out[len - i - 1] += current_sum_descending;
                current_sum_descending += rev_v_i;
                current_sum_ascending += v_i;
            });
    }

    out.iter_mut().for_each(|x| *x /= len as f64);
}

pub fn grand_means_weighted(v: &[f64], w: &[f64], out: &mut [f64], len: usize) {
    let mut current_sum_ascending = 0.0;
    let mut current_sum_descending = 0.0;

    let mut sum = -w[1..].iter().sum::<f64>();
    (0..len).for_each(|i| {
        out[i] += sum * v[i] - current_sum_ascending;
        if i < len - 1 {
            sum += w[i] + w[i + 1];
        }
        let rev_ord_i = len - i - 1;
        out[rev_ord_i] += current_sum_descending;
        current_sum_descending += v[rev_ord_i] * w[rev_ord_i];
        current_sum_ascending += v[i] * w[i];
    });

    out.iter_mut().for_each(|x| *x /= len as f64);
}
