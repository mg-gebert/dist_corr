// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Modules

pub mod api;
pub(crate) mod dist_corr_binary;
pub(crate) mod dist_corr_fast;
pub(crate) mod dist_corr_multi;
pub(crate) mod dist_corr_naive;
pub(crate) mod frob_inner_product;
pub(crate) mod grand_mean;
pub(crate) mod tests;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Export

#[doc(inline)]
pub use api::{
    dist_corr, dist_corr_binary, dist_corr_one_binary, dist_cov, dist_cov_binary,
    dist_cov_one_binary, dist_var,
};
