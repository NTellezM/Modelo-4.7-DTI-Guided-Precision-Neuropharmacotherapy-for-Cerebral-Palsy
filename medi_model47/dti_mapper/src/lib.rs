// src/lib.rs
//! # dti_mapper
//!
//! High-performance log-Euclidean DTI tensor interpolation onto FEM meshes.
//!
//! This crate provides the computational core for mapping diffusion tensor imaging
//! (DTI) data from a regular voxel grid onto the nodes of an unstructured finite
//! element mesh. The log-Euclidean framework guarantees that interpolated tensors
//! remain symmetric positive definite (SPD).
//!
//! ## Python API
//!
//! All public functions are exposed to Python via PyO3. Import as:
//!
//! ```python
//! from dti_mapper._rust import (
//!     precompute_logs,
//!     interpolate_fem_points,
//!     compute_trilinear_weights,
//!     validate_tensors,
//!     tensor_log_single,
//!     tensor_exp_single,
//! )
//! ```

// Clippy 1.94+ flags `?` on Result<T, DtiError> in functions returning PyResult<T>
// as "useless conversion to the same type: pyo3::PyErr". This is a false positive:
// DtiError ≠ PyErr, and the From impl is the correct conversion path.
#![allow(clippy::useless_conversion)]

pub mod errors;
pub mod interpolation;
pub mod tensor_ops;
pub mod validation;

use numpy::ndarray::{Array2, ArrayView3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use nalgebra::Matrix3;

use crate::errors::DtiError;
use crate::tensor_ops::DEFAULT_EPSILON;

// ---------------------------------------------------------------------------
// Helper: convert between NumPy (N, 3, 3) and internal Vec<Matrix3<f64>>
// ---------------------------------------------------------------------------

/// Read a (N, 3, 3) array into Vec<Matrix3<f64>>.
fn ndarray_to_matrices(arr: ArrayView3<f64>) -> Vec<Matrix3<f64>> {
    let n = arr.shape()[0];
    (0..n)
        .map(|i| {
            Matrix3::new(
                arr[[i, 0, 0]],
                arr[[i, 0, 1]],
                arr[[i, 0, 2]],
                arr[[i, 1, 0]],
                arr[[i, 1, 1]],
                arr[[i, 1, 2]],
                arr[[i, 2, 0]],
                arr[[i, 2, 1]],
                arr[[i, 2, 2]],
            )
        })
        .collect()
}

/// Write Vec<Matrix3<f64>> into a (N, 9) ndarray (row-major flat tensors).
fn matrices_to_ndarray(matrices: &[Matrix3<f64>]) -> Array2<f64> {
    let n = matrices.len();
    let mut arr = Array2::zeros((n, 9));
    for (i, m) in matrices.iter().enumerate() {
        for r in 0..3 {
            for c in 0..3 {
                arr[[i, r * 3 + c]] = m[(r, c)];
            }
        }
    }
    arr
}

/// Flatten Vec<Matrix3<f64>> into a flat f64 slice for validation.
fn matrices_to_flat(matrices: &[Matrix3<f64>]) -> Vec<f64> {
    let mut flat = Vec::with_capacity(matrices.len() * 9);
    for m in matrices {
        for r in 0..3 {
            for c in 0..3 {
                flat.push(m[(r, c)]);
            }
        }
    }
    flat
}

// ---------------------------------------------------------------------------
// Python-exposed functions
// ---------------------------------------------------------------------------

/// Precompute matrix logarithms for a volume of DTI tensors.
///
/// Parameters
/// ----------
/// tensors : np.ndarray, shape (N, 3, 3), dtype float64
///     Input SPD tensors (e.g., flattened DTI volume).
/// epsilon : float, optional
///     Clamping threshold for small eigenvalues. Default: 1e-12.
///
/// Returns
/// -------
/// np.ndarray, shape (N, 9), dtype float64
///     Matrix logarithms in row-major flat format (for efficient indexing).
#[pyfunction]
#[pyo3(signature = (tensors, epsilon=None))]
fn precompute_logs<'py>(
    py: Python<'py>,
    tensors: PyReadonlyArray3<'py, f64>,
    epsilon: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let eps = epsilon.unwrap_or(DEFAULT_EPSILON);
    let arr = tensors.as_array();
    let n = arr.shape()[0];

    let matrices = ndarray_to_matrices(arr);
    let flat: Vec<f64> = matrices_to_flat(&matrices);

    let log_matrices = tensor_ops::precompute_logs_batch(&flat, n, eps)?;

    let result = matrices_to_ndarray(&log_matrices);
    Ok(result.into_pyarray_bound(py))
}

/// Interpolate pre-computed log-tensors at FEM node positions.
///
/// Parameters
/// ----------
/// log_tensors : np.ndarray, shape (N_voxels, 9), dtype float64
///     Pre-computed matrix logarithms from `precompute_logs`.
/// indices : np.ndarray, shape (N_nodes, 8), dtype int64
///     Trilinear neighbor indices for each FEM node.
/// weights : np.ndarray, shape (N_nodes, 8), dtype float64
///     Trilinear weights for each FEM node (rows sum to 1).
///
/// Returns
/// -------
/// np.ndarray, shape (N_nodes, 9), dtype float64
///     Interpolated SPD tensors at each FEM node (row-major flat).
#[pyfunction]
fn interpolate_fem_points_py<'py>(
    py: Python<'py>,
    log_tensors: PyReadonlyArray2<'py, f64>,
    indices: PyReadonlyArray2<'py, i64>,
    weights: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let log_arr = log_tensors.as_array();
    let idx_arr = indices.as_array();
    let w_arr = weights.as_array();

    let n_voxels = log_arr.shape()[0];
    let n_nodes = idx_arr.shape()[0];

    // Convert log_tensors (N, 9) → Vec<Matrix3>
    let log_matrices: Vec<Matrix3<f64>> = (0..n_voxels)
        .map(|i| Matrix3::from_row_slice(&(0..9).map(|j| log_arr[[i, j]]).collect::<Vec<f64>>()))
        .collect();

    // Flatten indices and weights
    let flat_indices: Vec<usize> = idx_arr
        .iter()
        .map(|&v| if v < 0 { 0usize } else { v as usize })
        .collect();
    let flat_weights: Vec<f64> = w_arr.iter().copied().collect();

    let result = interpolation::interpolate_fem_points(
        &log_matrices,
        &flat_indices,
        &flat_weights,
        n_nodes,
        n_voxels,
    )?;

    let result_arr = matrices_to_ndarray(&result);
    Ok(result_arr.into_pyarray_bound(py))
}

/// Compute trilinear interpolation indices and weights.
///
/// Parameters
/// ----------
/// points : np.ndarray, shape (N, 3), dtype float64
///     Physical coordinates of FEM nodes (in mm, RAS space).
/// inv_affine : np.ndarray, shape (4, 4), dtype float64
///     Inverse affine matrix (physical → voxel coordinates).
/// grid_shape : tuple of (int, int, int)
///     DTI volume dimensions (Nx, Ny, Nz).
///
/// Returns
/// -------
/// indices : np.ndarray, shape (N, 8), dtype int64
/// weights : np.ndarray, shape (N, 8), dtype float64
#[pyfunction]
#[allow(clippy::type_complexity)]
fn compute_trilinear_weights_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    inv_affine: PyReadonlyArray2<'py, f64>,
    grid_shape: (usize, usize, usize),
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
    let pts = points.as_array();
    let aff = inv_affine.as_array();
    let n_points = pts.shape()[0];

    let flat_points: Vec<f64> = pts.iter().copied().collect();

    let mut inv_aff = [0.0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            inv_aff[i * 4 + j] = aff[[i, j]];
        }
    }

    let (indices, weights) =
        interpolation::compute_trilinear_weights(&flat_points, n_points, &inv_aff, grid_shape)?;

    let idx_arr =
        Array2::from_shape_vec((n_points, 8), indices.iter().map(|&i| i as i64).collect())
            .map_err(|e| DtiError::NumpyError(e.to_string()))?;

    let w_arr = Array2::from_shape_vec((n_points, 8), weights)
        .map_err(|e| DtiError::NumpyError(e.to_string()))?;

    Ok((idx_arr.into_pyarray_bound(py), w_arr.into_pyarray_bound(py)))
}

/// Validate a batch of tensors for symmetry, positive definiteness, and finiteness.
///
/// Parameters
/// ----------
/// tensors : np.ndarray, shape (N, 3, 3), dtype float64
/// symmetry_tol : float, optional
///     Tolerance for symmetry check. Default: 1e-8.
///
/// Returns
/// -------
/// dict with keys:
///     n_total, n_symmetric, n_spd, n_finite, min_eigenvalue, max_eigenvalue,
///     mean_eigenvalue, mean_fa, min_fa, max_fa, max_symmetry_error, all_valid
#[pyfunction]
#[pyo3(signature = (tensors, symmetry_tol=None))]
fn validate_tensors_py<'py>(
    py: Python<'py>,
    tensors: PyReadonlyArray3<'py, f64>,
    symmetry_tol: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let tol = symmetry_tol.unwrap_or(tensor_ops::DEFAULT_SYMMETRY_TOL);
    let arr = tensors.as_array();
    let n = arr.shape()[0];

    let matrices = ndarray_to_matrices(arr);
    let flat = matrices_to_flat(&matrices);

    let report = validation::validate_tensors(&flat, n, tol);

    let dict = PyDict::new_bound(py);
    dict.set_item("n_total", report.n_total)?;
    dict.set_item("n_symmetric", report.n_symmetric)?;
    dict.set_item("n_spd", report.n_spd)?;
    dict.set_item("n_finite", report.n_finite)?;
    dict.set_item("min_eigenvalue", report.min_eigenvalue)?;
    dict.set_item("max_eigenvalue", report.max_eigenvalue)?;
    dict.set_item("mean_eigenvalue", report.mean_eigenvalue)?;
    dict.set_item("mean_fa", report.mean_fa)?;
    dict.set_item("min_fa", report.min_fa)?;
    dict.set_item("max_fa", report.max_fa)?;
    dict.set_item("max_symmetry_error", report.max_symmetry_error)?;
    dict.set_item("all_valid", report.all_valid())?;
    dict.set_item("summary", report.summary())?;
    Ok(dict)
}

/// Compute the matrix logarithm of a single 3×3 SPD tensor.
///
/// Useful for debugging and testing.
#[pyfunction]
#[pyo3(signature = (tensor, epsilon=None))]
fn tensor_log_single<'py>(
    py: Python<'py>,
    tensor: PyReadonlyArray2<'py, f64>,
    epsilon: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let eps = epsilon.unwrap_or(DEFAULT_EPSILON);
    let arr = tensor.as_array();

    let m = Matrix3::new(
        arr[[0, 0]],
        arr[[0, 1]],
        arr[[0, 2]],
        arr[[1, 0]],
        arr[[1, 1]],
        arr[[1, 2]],
        arr[[2, 0]],
        arr[[2, 1]],
        arr[[2, 2]],
    );

    let log_m = tensor_ops::tensor_log(&m, eps)?;

    let mut result = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            result[[i, j]] = log_m[(i, j)];
        }
    }

    Ok(result.into_pyarray_bound(py))
}

/// Compute the matrix exponential of a single 3×3 symmetric tensor.
#[pyfunction]
fn tensor_exp_single<'py>(
    py: Python<'py>,
    tensor: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = tensor.as_array();

    let m = Matrix3::new(
        arr[[0, 0]],
        arr[[0, 1]],
        arr[[0, 2]],
        arr[[1, 0]],
        arr[[1, 1]],
        arr[[1, 2]],
        arr[[2, 0]],
        arr[[2, 1]],
        arr[[2, 2]],
    );

    let exp_m = tensor_ops::tensor_exp(&m)?;

    let mut result = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            result[[i, j]] = exp_m[(i, j)];
        }
    }

    Ok(result.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Python module definition.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(precompute_logs, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_fem_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_trilinear_weights_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tensors_py, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_log_single, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_exp_single, m)?)?;

    m.add("DEFAULT_EPSILON", DEFAULT_EPSILON)?;
    m.add("DEFAULT_SYMMETRY_TOL", tensor_ops::DEFAULT_SYMMETRY_TOL)?;

    Ok(())
}
