use itertools::izip;
use rayon::join;

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

pub fn compute_frobenius_inner_product(samples0: &[f64], samples1: &[f64], len: usize) -> f64 {
    // initialize buffered indices
    let mut idxs_before: Vec<usize> = (0..len).collect();
    let mut idxs_after: Vec<usize> = vec![0; len];

    let middle = (len as f64 / 2.0).ceil() as usize;

    let mut ivs = vec![Iv::default(); len];
    let mut csums = vec![Csum::default(); len + 1];
    let mut csums_left = vec![Csum::default(); middle + 1];
    let mut csums_right = vec![Csum::default(); len - middle + 1];

    let mut idxs_before_left = idxs_before[..middle].to_vec();
    let mut idxs_after_left = idxs_after[..middle].to_vec();

    let mut idxs_before_right = idxs_before[middle..].to_vec();
    let mut idxs_after_right = idxs_after[middle..].to_vec();

    let mut ivs_left = ivs.clone();
    let mut ivs_right = ivs.clone();

    let ((), ()) = join(
        || {
            perform_loop(
                samples0,
                samples1,
                &mut idxs_before_left,
                &mut idxs_after_left,
                middle,
                &mut 1,
                &mut csums_left,
                &mut ivs_left,
            );
        },
        || {
            perform_loop(
                samples0,
                samples1,
                &mut idxs_before_right,
                &mut idxs_after_right,
                len - middle,
                &mut 1,
                &mut csums_right,
                &mut ivs_right,
            );
        },
    );

    /*
        perform_loop(
            samples0,
            samples1,
            &mut idxs_before_left,
            &mut idxs_after_left,
            middle,
            &mut 1,
            &mut csums_left,
            &mut ivs,
        );

        perform_loop(
            samples0,
            samples1,
            &mut idxs_before_right,
            &mut idxs_after_right,
            len - middle,
            &mut 1,
            &mut csums_right,
            &mut ivs,
        );
    */

    idxs_after[..middle]
        .iter_mut()
        .zip(idxs_after_left)
        .for_each(|(x, y)| *x = y);
    idxs_after[middle..]
        .iter_mut()
        .zip(idxs_after_right)
        .for_each(|(x, y)| *x = y);

    idxs_before[..middle]
        .iter_mut()
        .zip(idxs_before_left)
        .for_each(|(x, y)| *x = y);
    idxs_before[middle..]
        .iter_mut()
        .zip(idxs_before_right)
        .for_each(|(x, y)| *x = y);

    ivs[..middle]
        .iter_mut()
        .zip(ivs_left[..middle].iter())
        .for_each(|(x, y)| *x = y.clone());
    ivs[middle..]
        .iter_mut()
        .zip(ivs_right[middle..].iter())
        .for_each(|(x, y)| *x = y.clone());

    perform_loop(
        samples0,
        samples1,
        &mut idxs_before,
        &mut idxs_after,
        len,
        &mut middle.clone(),
        &mut csums,
        &mut ivs,
    );

    let cov_term = len as f64 * csums[len].xy - csums[len].x * csums[len].y;

    let sum = izip!(ivs, samples0, samples1)
        .map(|(iv, s0, s1)| 4.0 * (iv.num as f64 * s0 * s1 + iv.xy - iv.x * s0 - iv.y * s1))
        .sum::<f64>();

    sum - 2.0 * cov_term
}

fn perform_loop(
    samples0: &[f64],
    samples1: &[f64],
    idxs_before: &mut [usize],
    idxs_after: &mut [usize],
    len: usize,
    i: &mut usize,
    csums: &mut [Csum],
    ivs: &mut [Iv],
) {
    while *i < len {
        // update cum sums
        (0..len).for_each(|ind| {
            let (x, y) = (samples1[idxs_before[ind]], samples0[idxs_before[ind]]);

            csums[ind + 1].x = x + csums[ind].x;
            csums[ind + 1].y = y + csums[ind].y;
            csums[ind + 1].xy = x * y + csums[ind].xy;
        });

        izip!(
            (0..len).step_by(2 * *i),
            idxs_before.chunks(2 * *i),
            idxs_after.chunks_mut(2 * *i),
        )
        .for_each(|(j, idx_r_j, idx_s_j)| {
            let mut k = 0;
            let mut st1 = 0;
            let mut st2 = *i;

            let e1_abs = len.min(j + *i);
            let e1_rel = (e1_abs - j).max(0);

            let e2_rel = (len.min(2 * *i + j) - j).max(0);

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
        *i *= 2;

        idxs_before
            .iter_mut()
            .zip(idxs_after.iter())
            .for_each(|(r, s)| *r = *s);
    }
}
