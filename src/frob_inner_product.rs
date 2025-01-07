use itertools::izip;

pub fn compute_frobenius_inner_product(samples0: &[f64], samples1: &[f64], len: usize) -> f64 {
    // initialize buffered indices
    let mut idxs_before: Vec<usize> = (0..len).collect();
    let mut idxs_after: Vec<usize> = vec![0; len];

    #[derive(Clone, Default)]
    struct Iv {
        num: usize,
        x: f64,
        y: f64,
        xy: f64,
    }

    #[derive(Clone, Default)]
    struct Csum {
        x: f64,
        y: f64,
        xy: f64,
    }

    let mut ivs = vec![Iv::default(); len];
    let mut csums = vec![Csum::default(); len + 1];

    // init vars for buffering and sorting
    let mut i = 1;
    let mut cov_term = 0.0;

    // do iteration
    while i < len {
        // update cum sums
        (0..len).for_each(|ind| {
            let (x, y) = (samples1[idxs_before[ind]], samples0[idxs_before[ind]]);

            csums[ind + 1].x = x + csums[ind].x;
            csums[ind + 1].y = y + csums[ind].y;
            csums[ind + 1].xy = x * y + csums[ind].xy;
        });

        if i == 1 {
            cov_term = len as f64 * csums[len].xy - csums[len].x * csums[len].y;
        }

        let gap = 2 * i;

        izip!(
            (0..len).step_by(gap),
            idxs_before.chunks(gap),
            idxs_after.chunks_mut(gap),
        )
        .for_each(|(j, idx_r_j, idx_s_j)| {
            let mut k = 0;
            let mut st1 = 0;
            let mut st2 = i;

            let e1_abs = len.min(j + i);
            let e1_rel = (e1_abs - j).max(0);

            let e2_rel = (len.min(2 * i + j) - j).max(0);

            while e1_rel > st1 && e2_rel > st2 {
                let idx1 = idx_r_j[st1];
                let idx2 = idx_r_j[st2];
                if samples0[idx1] >= samples0[idx2] {
                    st1 += 1;
                    idx_s_j[k] = idx1;
                } else {
                    idx_s_j[k] = idx2;
                    st2 += 1;

                    ivs[idx2].num += e1_rel - st1;
                    ivs[idx2].x += csums[e1_abs].x - csums[j + st1].x;
                    ivs[idx2].y += csums[e1_abs].y - csums[j + st1].y;
                    ivs[idx2].xy += csums[e1_abs].xy - csums[j + st1].xy;
                }

                k += 1;
            }

            if e1_rel > st1 {
                idx_s_j[k..k + e1_rel - st1].copy_from_slice(&idx_r_j[st1..e1_rel]);
            } else if e2_rel > st2 {
                idx_s_j[k..k + e2_rel - st2].copy_from_slice(&idx_r_j[st2..e2_rel]);
            }
        });
        i = gap;

        idxs_before
            .iter_mut()
            .zip(idxs_after.iter())
            .for_each(|(r, s)| *r = *s);
    }

    let sum = izip!(ivs, samples0, samples1)
        .map(|(iv, s0, s1)| 4.0 * (iv.num as f64 * s0 * s1 + iv.xy - iv.x * s0 - iv.y * s1))
        .sum::<f64>();

    sum - 2.0 * cov_term
}
