//! dist_corr — distance correlation utilities
//!
//! Distance-correlation routines for 1D data.
//!
//! # Quick example
//!
//! ```
//! use dist_corr::DistCorrelation;
//!
//! let v1 = vec![1.0, 2.0, 3.0];
//! let v2 = vec![2.0, 4.0, 6.0];
//! let dist_corr = DistCorrelation;
//! let result = dist_corr.compute(&v1, &v2);
//! println!("{:?}", result);
//!
//! // for one vector being binary, i.e. 0-1 valued, a faster algorithm is implemented.
//! let v1 = vec![1.0, 1.0, 0.0];
//! let v2 = vec![2.0, 4.0, 6.0];
//! let dist_corr = DistCorrelation;
//! let result = dist_corr.compute_binary(&v1, &v2, true, false);
//! println!("{:?}", result);
//!
//! // for both vectors being binary, an even faster algorithm is implemented
//! let v1 = vec![1.0, 1.0, 0.0, 0.0];
//! let v2 = vec![1.0, 0.0, 1.0, 0.0];
//! let dist_corr = DistCorrelation;
//! let result = dist_corr.compute_binary(&v1, &v2, true, true);
//! println!("{:?}", result);
//! ```
//!
//! # Theory
//!
//! For
//! See the [distance correlation](dist_corr_fast_short.pdf) for more details.

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Modules

pub mod api;
pub(crate) mod dist_corr;
pub(crate) mod dist_corr_binary;
pub(crate) mod dist_corr_multi;
pub(crate) mod dist_corr_naive;
pub(crate) mod frob_inner_product;
pub(crate) mod grand_mean;
pub(crate) mod ordering;
pub(crate) mod tests;

// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Export

#[doc(inline)]
pub use api::DistCorrelation;
#[doc(inline)]
pub use api::DistCovariance;
