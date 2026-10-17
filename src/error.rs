// +++++++++++++++++++++++++++++++++++++++++++++++++++
// Implementation

/// Identifies which input vector an error refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vector {
    V1,
    V2,
}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Vector::V1 => write!(f, "v1"),
            Vector::V2 => write!(f, "v2"),
        }
    }
}

/// Error type for computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    VectorLengthMismatch,
    VectorNotBinary(Vector),
    VectorEmpty,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::VectorLengthMismatch => write!(f, "Vector lengths must be identical"),
            Error::VectorNotBinary(v) => {
                write!(f, "Vector {v} must be binary (only 0.0 or 1.0)")
            }
            Error::VectorEmpty => write!(f, "Vector must not be empty"),
        }
    }
}

impl std::error::Error for Error {}
