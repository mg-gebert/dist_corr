// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Modules

pub(crate) mod api;
pub(crate) mod dist_corr_fast;
pub(crate) mod dist_corr_naive;
pub(crate) mod frob_inner_product;
pub(crate) mod grand_mean;
pub(crate) mod tests;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Export

#[doc(inline)]
pub use api::{dist_corr, dist_cov, dist_var};
