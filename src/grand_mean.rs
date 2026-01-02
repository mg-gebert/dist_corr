// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Using

use itertools::izip;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Struct

pub struct GrandMeans<'a> {
    v: &'a [f64],
    len: usize,
    out: Vec<f64>,
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

/// Grand means provider
impl<'a> GrandMeans<'a> {
    /// Create a new GrandMeans provider instance.
    pub fn new(v: &'a [f64]) -> Self {
        let len = v.len();
        let out = vec![0.0; len];

        Self { v, len, out }
    }

    /// Computes the grand means of the matrix
    ///
    /// ```text
    /// M_v[i][j] = |v[i] - v[j]|
    /// ```
    ///
    /// for the vector `v = self.v` where we assume that
    /// v is ordered increasingly.
    /// The grand mean for each index `i` is defined as:
    ///
    /// ```text
    /// GM[i] =  sum_j |v[i] - v[j]| / v.len()
    /// ```
    ///
    /// The resulting grand means are written into `self.out`.
    /// The algorithm has complexity `O(v.len())`.
    pub fn compute_ordered(&mut self) {
        let mut current_sum_ascending = 0.0;
        let mut current_sum_descending = 0.0;

        izip!(self.v.iter(), self.v.iter().rev())
            .enumerate()
            .for_each(|(i, (v_i, rev_v_i))| {
                self.out[i] +=
                    (2 * (i as i64) - self.len as i64 + 1) as f64 * v_i - current_sum_ascending;
                self.out[self.len - i - 1] += current_sum_descending;
                current_sum_descending += rev_v_i;
                current_sum_ascending += v_i;
            });

        self.out.iter_mut().for_each(|x| *x /= self.len as f64);
    }

    /// Computes the grand means of the matrix
    ///
    /// ```text
    /// M_v[i][j] = |v[i] - v[j]|
    /// ```
    ///
    /// for the vector `v = self.v` where `order` denotes
    /// the indices to order v increasingly.
    /// The grand mean for each index `i` is defined as:
    ///
    /// ```text
    /// GM[i] =  sum_j |v[i] - v[j]| / v.len()
    /// ```
    ///
    /// The resulting grand means are written into `self.out`.
    /// The algorithm has complexity `O(v.len())`.
    pub fn compute_unordered(&mut self, order: &[usize]) {
        assert_eq!(order.len(), self.len, "order must be same length as v");
        let mut current_sum_ascending = 0.0;
        let mut current_sum_descending = 0.0;

        izip!(order.iter(), order.iter().rev())
            .enumerate()
            .for_each(|(i, (ord_j, rev_ord_j))| {
                self.out[*ord_j] += (2 * (i as i64) - self.len as i64 + 1) as f64 * self.v[*ord_j]
                    - current_sum_ascending;
                self.out[*rev_ord_j] += current_sum_descending;
                current_sum_descending += self.v[*rev_ord_j];
                current_sum_ascending += self.v[*ord_j];
            });

        self.out.iter_mut().for_each(|x| *x /= self.len as f64);
    }

    /// Computes the matrix multiplication
    ///
    /// ```text
    /// M_v*w / v.len()
    /// ```
    /// where
    /// ```text
    /// M_v[i][j] = |v[i] - v[j]|
    /// ```
    ///
    /// The resulting grand means are written into `self.out`.
    /// The algorithm has complexity `O(v.len())`.
    pub fn compute_ordered_weighted(&mut self, w: &[f64]) {
        assert_eq!(w.len(), self.len, "weights must be same length as v");
        let mut current_sum_ascending = 0.0;
        let mut current_sum_descending = 0.0;

        let mut sum = -w[1..].iter().sum::<f64>();
        for i in 0..self.len {
            self.out[i] += sum * self.v[i] - current_sum_ascending;
            if i < self.len - 1 {
                sum += w[i] + w[i + 1];
            }
            let rev_ord_i = self.len - i - 1;
            self.out[rev_ord_i] += current_sum_descending;
            current_sum_descending += self.v[rev_ord_i] * w[rev_ord_i];
            current_sum_ascending += self.v[i] * w[i];
        }

        self.out.iter_mut().for_each(|x| *x /= self.len as f64);
    }

    /// returns the grand means stored in `self.out`.
    ///
    /// Note that `compute_ordered`, `compute_ordered_weighted`,  `compute_unordered`
    /// should have been called before.
    pub(crate) fn get_means(&self) -> &[f64] {
        &self.out
    }
}
