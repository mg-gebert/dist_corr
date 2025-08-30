use itertools::izip;
// Import
use rayon::prelude::*;

// move
pub fn dist_cov_multi(v_1: &[Vec<f64>], v_2: &[Vec<f64>]) -> f64 {
    let data_length = v_1.len() as f64;
    let dist_frob_norm = v_1
        .par_iter()
        .zip(v_2.par_iter())
        .enumerate()
        .map(|(i, (a1, b1))| {
            v_1[i..]
                .iter()
                .zip(v_2[i..].iter())
                .map(|(a2, b2)| {
                    izip!(a1, a2)
                        .map(|(a1_i, a2_i)| ((a1_i - a2_i).powi(2)))
                        .sum::<f64>()
                        .sqrt()
                        * izip!(b1, b2)
                            .map(|(b1_i, b2_i)| ((b1_i - b2_i).powi(2)))
                            .sum::<f64>()
                            .sqrt()
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        * 2.0
        / (data_length * data_length);

    let dist_scalar_avg = v_1
        .par_iter()
        .zip(v_2.par_iter())
        .map(|(a1, b1)| {
            let data1i = v_1
                .iter()
                .map(|a2| {
                    izip!(a1, a2)
                        .map(|(a1_i, a2_i)| ((a1_i - a2_i).powi(2)))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>();
            let data2i = v_2
                .iter()
                .map(|b2| {
                    izip!(b1, b2)
                        .map(|(b1_i, b2_i)| ((b1_i - b2_i).powi(2)))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>();
            data1i * data2i
        })
        .sum::<f64>()
        / (data_length * data_length * data_length);

    let mut mean_a = 0.0;
    let mut mean_b = 0.0;
    v_1.iter().zip(v_2.iter()).for_each(|(a1, b1)| {
        mean_a += v_1
            .iter()
            .map(|a2| {
                izip!(a1, a2)
                    .map(|(a1_i, a2_i)| ((a1_i - a2_i).powi(2)))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum::<f64>();
        mean_b += v_2
            .iter()
            .map(|b2| {
                izip!(b1, b2)
                    .map(|(b1_i, b2_i)| ((b1_i - b2_i).powi(2)))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum::<f64>();
    });

    let means = mean_a * mean_b / (data_length * data_length * data_length * data_length);

    dist_frob_norm - 2.0 * dist_scalar_avg + means
}

// move
pub fn dist_cov_multi_exp(v_1: &[Vec<f64>], v_2: &[Vec<f64>]) -> f64 {
    let data_length = v_1.len() as f64;
    let dist_frob_norm = (v_1
        .par_iter()
        .zip(v_2.par_iter())
        .enumerate()
        .map(|(i, (a1, b1))| {
            v_1[i..]
                .iter()
                .zip(v_2[i..].iter())
                .map(|(a2, b2)| {
                    izip!(a1, a2)
                        .map(|(a1_i, a2_i)| (-(a1_i - a2_i).abs()).exp())
                        .product::<f64>()
                        * izip!(b1, b2)
                            .map(|(b1_i, b2_i)| (-(b1_i - b2_i).abs()).exp())
                            .product::<f64>()
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        * 2.0
        - data_length)
        / (data_length * data_length);

    let dist_scalar_avg = v_1
        .par_iter()
        .zip(v_2.par_iter())
        .map(|(a1, b1)| {
            let data1i = v_1
                .iter()
                .map(|a2| {
                    izip!(a1, a2)
                        .map(|(a1_i, a2_i)| (-(a1_i - a2_i).abs()).exp())
                        .product::<f64>()
                })
                .sum::<f64>();
            let data2i = v_2
                .iter()
                .map(|b2| {
                    izip!(b1, b2)
                        .map(|(b1_i, b2_i)| (-(b1_i - b2_i).abs()).exp())
                        .product::<f64>()
                })
                .sum::<f64>();
            data1i * data2i
        })
        .sum::<f64>()
        / (data_length * data_length * data_length);

    let mut mean_a = 0.0;
    let mut mean_b = 0.0;
    v_1.iter().zip(v_2.iter()).for_each(|(a1, b1)| {
        mean_a += v_1
            .iter()
            .map(|a2| {
                izip!(a1, a2)
                    .map(|(a1_i, a2_i)| (-(a1_i - a2_i).abs()).exp())
                    .product::<f64>()
            })
            .sum::<f64>();
        mean_b += v_2
            .iter()
            .map(|b2| {
                izip!(b1, b2)
                    .map(|(b1_i, b2_i)| (-(b1_i - b2_i).abs()).exp())
                    .product::<f64>()
            })
            .sum::<f64>();
    });

    let means = mean_a * mean_b / (data_length * data_length * data_length * data_length);

    dist_frob_norm - 2.0 * dist_scalar_avg + means
}
