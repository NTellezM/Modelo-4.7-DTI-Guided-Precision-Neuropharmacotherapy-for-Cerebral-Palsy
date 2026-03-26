// src/errors.rs
//! Typed error handling for the dti_mapper crate.
//!
//! All errors are convertible to Python exceptions via PyO3.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

/// Top-level error type for all dti_mapper operations.
#[derive(Error, Debug)]
pub enum DtiError {
    #[error("Tensor is not symmetric: ||T - T^T|| = {norm:.2e} > tolerance {tol:.2e}")]
    NotSymmetric { norm: f64, tol: f64 },

    #[error("Tensor is not positive definite: min eigenvalue = {min_ev:.2e}")]
    NotPositiveDefinite { min_ev: f64 },

    #[error("Eigenvalue too small for log: {value:.2e} < epsilon {epsilon:.2e}")]
    EigenvalueTooSmall { value: f64, epsilon: f64 },

    #[error("Dimension mismatch: expected {expected}, got {got} ({context})")]
    DimensionMismatch {
        expected: String,
        got: String,
        context: String,
    },

    #[error("Index out of bounds: index {index} for grid of size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Interpolation weights do not sum to 1: sum = {sum:.6e}")]
    WeightSumError { sum: f64 },

    #[error("NaN or Inf detected in tensor at position {position}")]
    NonFinite { position: usize },

    #[error("NumPy array error: {0}")]
    NumpyError(String),
}

impl From<DtiError> for PyErr {
    fn from(err: DtiError) -> PyErr {
        match &err {
            DtiError::DimensionMismatch { .. }
            | DtiError::IndexOutOfBounds { .. }
            | DtiError::NumpyError(_) => PyValueError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

/// Convenience type alias.
pub type DtiResult<T> = Result<T, DtiError>;
