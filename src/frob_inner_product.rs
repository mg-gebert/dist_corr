// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use itertools::izip;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Definition

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

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

pub fn compute_frobenius_inner_product(samples0: &[f64], samples1: &[f64], len: usize) -> f64 {
    // initialize indices
    let mut idxs_before: Vec<usize> = (0..len).collect();
    let mut idxs_after: Vec<usize> = vec![0; len];

    let num_threads = rayon::current_num_threads();
    println!("Number of available threads in Rayon: {}", num_threads);

    let chunk_size = (len as f64 / num_threads as f64).ceil() as usize;

    let mut ivs = vec![Iv::default(); len];
    let mut csums = vec![Csum::default(); len + num_threads];

    idxs_before
        .par_chunks_mut(chunk_size)
        .zip(idxs_after.par_chunks_mut(chunk_size))
        .zip(ivs.par_chunks_mut(chunk_size))
        .zip(csums.par_chunks_mut(chunk_size + 1))
        .enumerate()
        .for_each(
            |(j, (((idxs_before_chunk, idxs_after_chunk), ivs_chunk), csums_chunk))| {
                perform_loop(
                    samples0,
                    samples1,
                    idxs_before_chunk,
                    idxs_after_chunk,
                    ivs_chunk.len(),
                    &mut 1,
                    j * chunk_size,
                    csums_chunk,
                    ivs_chunk,
                );
            },
        );

    perform_loop(
        samples0,
        samples1,
        &mut idxs_before,
        &mut idxs_after,
        len,
        &mut chunk_size.clone(),
        0,
        &mut csums[..len + 1],
        &mut ivs,
    );

    let cov_term = len as f64 * csums[len].xy - csums[len].x * csums[len].y;

    let sum = izip!(ivs, samples0, samples1)
        .map(|(iv, s0, s1)| 4.0 * (iv.num as f64 * s0 * s1 + iv.xy - iv.x * s0 - iv.y * s1))
        .sum::<f64>();

    sum - 2.0 * cov_term
}

#[allow(clippy::too_many_arguments)]
fn perform_loop(
    samples0: &[f64],
    samples1: &[f64],
    idxs_before: &mut [usize],
    idxs_after: &mut [usize],
    len: usize,
    idx_start: &mut usize,
    first_index: usize,
    csums: &mut [Csum],
    ivs: &mut [Iv],
) {
    while *idx_start < len {
        // update cum sums
        (0..len).for_each(|ind| {
            let (x, y) = (samples1[idxs_before[ind]], samples0[idxs_before[ind]]);

            csums[ind + 1].x = x + csums[ind].x;
            csums[ind + 1].y = y + csums[ind].y;
            csums[ind + 1].xy = x * y + csums[ind].xy;
        });

        izip!(
            (0..len).step_by(2 * *idx_start),
            idxs_before.chunks(2 * *idx_start),
            idxs_after.chunks_mut(2 * *idx_start),
        )
        .for_each(|(j, idx_r_j, idx_s_j)| {
            let mut k = 0;
            let mut st1 = 0;
            let mut st2 = *idx_start;

            let e1_abs = len.min(j + *idx_start);
            let e1_rel = (e1_abs - j).max(0);

            let e2_rel = (len.min(2 * *idx_start + j) - j).max(0);

            while e1_rel > st1 && e2_rel > st2 {
                let idx1 = idx_r_j[st1];
                let idx2 = idx_r_j[st2];
                if samples0[idx1] >= samples0[idx2] {
                    st1 += 1;
                    idx_s_j[k] = idx1;
                } else {
                    idx_s_j[k] = idx2;
                    st2 += 1;

                    let idx2_eff = idx2 - first_index;

                    ivs[idx2_eff].num += e1_rel - st1;
                    ivs[idx2_eff].x += csums[e1_abs].x - csums[j + st1].x;
                    ivs[idx2_eff].y += csums[e1_abs].y - csums[j + st1].y;
                    ivs[idx2_eff].xy += csums[e1_abs].xy - csums[j + st1].xy;
                }

                k += 1;
            }

            if e1_rel > st1 {
                idx_s_j[k..k + e1_rel - st1].copy_from_slice(&idx_r_j[st1..e1_rel]);
            } else if e2_rel > st2 {
                idx_s_j[k..k + e2_rel - st2].copy_from_slice(&idx_r_j[st2..e2_rel]);
            }
        });
        *idx_start *= 2;

        idxs_before
            .iter_mut()
            .zip(idxs_after.iter())
            .for_each(|(r, s)| *r = *s);
    }
}
