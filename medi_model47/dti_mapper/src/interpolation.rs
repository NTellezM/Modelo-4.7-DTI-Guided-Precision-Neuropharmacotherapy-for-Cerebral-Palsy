// src/interpolation.rs
//! Trilinear interpolation of DTI tensors onto FEM mesh nodes using the
//! log-Euclidean framework.
//!
//! The log-Euclidean metric treats the space of SPD matrices as a Riemannian
//! manifold. Interpolation in log-space (where SPD matrices live in a flat
//! vector space) guarantees that the result is always SPD — unlike naive
//! Euclidean interpolation which can produce non-positive-definite tensors.
//!
//! # Algorithm
//!
//! For each FEM node at physical position **p**:
//!
//! 1. Convert **p** to continuous voxel coordinates using the inverse affine.
//! 2. Identify the 8 surrounding voxels (trilinear stencil).
//! 3. Compute trilinear weights (products of 1D linear basis functions).
//! 4. Combine pre-computed log-tensors: `L = Σᵢ wᵢ * log(Tᵢ)`.
//! 5. Exponentiate: `T_interp = exp(L)`.
//!
//! Steps 4–5 are parallelised over FEM nodes using rayon.

use nalgebra::Matrix3;
use rayon::prelude::*;

use crate::errors::{DtiError, DtiResult};
use crate::tensor_ops::tensor_exp;

/// Maximum number of trilinear neighbors (2³ = 8 corners of the voxel cube).
pub const N_TRILINEAR: usize = 8;

/// Interpolate pre-computed log-tensors at FEM node positions.
///
/// This is the hot path of the entire pipeline. For a typical mesh of 500k nodes,
/// this performs ~500k weighted sums of 8 Matrix3 each, followed by 500k spectral
/// decompositions (tensor_exp). Rayon distributes across all available cores.
///
/// # Arguments
///
/// * `log_tensors` - Flat array of pre-computed matrix logarithms, indexed linearly
///   (row-major voxel grid flattened). Length = `grid_size` tensors.
/// * `indices` - For each FEM node, 8 linear indices into `log_tensors`.
///   Shape: `(n_fem_nodes, 8)`. Flat row-major: `indices[node * 8 + corner]`.
/// * `weights` - Trilinear weights for each FEM node.
///   Shape: `(n_fem_nodes, 8)`. Each row sums to 1.0.
/// * `n_nodes` - Number of FEM nodes.
/// * `grid_size` - Total number of voxels in the log_tensors grid.
///
/// # Returns
///
/// Vec of `n_nodes` interpolated SPD tensors (Matrix3<f64>).
///
/// # Errors
///
/// - `DimensionMismatch` if array lengths are inconsistent.
/// - `IndexOutOfBounds` if any neighbor index exceeds `grid_size`.
/// - `NonFinite` if any intermediate result contains NaN/Inf.
pub fn interpolate_fem_points(
    log_tensors: &[Matrix3<f64>],
    indices: &[usize],
    weights: &[f64],
    n_nodes: usize,
    grid_size: usize,
) -> DtiResult<Vec<Matrix3<f64>>> {
    // Validate dimensions
    if indices.len() != n_nodes * N_TRILINEAR {
        return Err(DtiError::DimensionMismatch {
            expected: format!("{}", n_nodes * N_TRILINEAR),
            got: format!("{}", indices.len()),
            context: "indices array".into(),
        });
    }
    if weights.len() != n_nodes * N_TRILINEAR {
        return Err(DtiError::DimensionMismatch {
            expected: format!("{}", n_nodes * N_TRILINEAR),
            got: format!("{}", weights.len()),
            context: "weights array".into(),
        });
    }

    // Pre-check: all indices in bounds
    if let Some(&max_idx) = indices.iter().max() {
        if max_idx >= grid_size {
            return Err(DtiError::IndexOutOfBounds {
                index: max_idx,
                size: grid_size,
            });
        }
    }

    // Parallel interpolation
    let results: Vec<DtiResult<Matrix3<f64>>> = (0..n_nodes)
        .into_par_iter()
        .map(|node| {
            let base = node * N_TRILINEAR;
            let mut sum = Matrix3::<f64>::zeros();

            for corner in 0..N_TRILINEAR {
                let idx = indices[base + corner];
                let w = weights[base + corner];

                if w.abs() < 1e-15 {
                    continue;
                }

                sum += log_tensors[idx] * w;
            }

            tensor_exp(&sum)
        })
        .collect();

    // Propagate first error
    results.into_iter().collect()
}

/// Compute trilinear interpolation indices and weights for a set of physical
/// coordinates within a regular voxel grid.
///
/// # Arguments
///
/// * `points` - Physical coordinates of FEM nodes, flat (n, 3) row-major.
/// * `n_points` - Number of points.
/// * `inv_affine` - 4×4 inverse affine matrix (physical → voxel), row-major [f64; 16].
/// * `grid_shape` - (Nx, Ny, Nz) dimensions of the DTI volume.
///
/// # Returns
///
/// Tuple of (indices, weights), each as flat Vec of length `n_points * 8`.
///
/// Points outside the grid are clamped to the nearest boundary voxel.
pub fn compute_trilinear_weights(
    points: &[f64],
    n_points: usize,
    inv_affine: &[f64; 16],
    grid_shape: (usize, usize, usize),
) -> DtiResult<(Vec<usize>, Vec<f64>)> {
    if points.len() != n_points * 3 {
        return Err(DtiError::DimensionMismatch {
            expected: format!("{}", n_points * 3),
            got: format!("{}", points.len()),
            context: "points array".into(),
        });
    }

    let (nx, ny, nz) = grid_shape;
    let mut all_indices = vec![0usize; n_points * N_TRILINEAR];
    let mut all_weights = vec![0.0f64; n_points * N_TRILINEAR];

    // This can also be parallelised, but it's typically fast enough serial
    // since it's just arithmetic (no eigen decompositions).
    for p in 0..n_points {
        let px = points[p * 3];
        let py = points[p * 3 + 1];
        let pz = points[p * 3 + 2];

        // Apply inverse affine: voxel = M^{-1} * [px, py, pz, 1]^T
        let vx = inv_affine[0] * px + inv_affine[1] * py + inv_affine[2] * pz + inv_affine[3];
        let vy = inv_affine[4] * px + inv_affine[5] * py + inv_affine[6] * pz + inv_affine[7];
        let vz = inv_affine[8] * px + inv_affine[9] * py + inv_affine[10] * pz + inv_affine[11];

        // Clamp to valid range [0, N-2] so floor and ceil are both valid indices
        let vx = vx.clamp(0.0, (nx as f64) - 1.0 - 1e-10);
        let vy = vy.clamp(0.0, (ny as f64) - 1.0 - 1e-10);
        let vz = vz.clamp(0.0, (nz as f64) - 1.0 - 1e-10);

        let ix0 = vx.floor() as usize;
        let iy0 = vy.floor() as usize;
        let iz0 = vz.floor() as usize;

        let ix1 = (ix0 + 1).min(nx - 1);
        let iy1 = (iy0 + 1).min(ny - 1);
        let iz1 = (iz0 + 1).min(nz - 1);

        // Fractional distances
        let fx = vx - ix0 as f64;
        let fy = vy - iy0 as f64;
        let fz = vz - iz0 as f64;

        // 8 corner indices (row-major C order: x varies slowest)
        let base = p * N_TRILINEAR;
        let corners = [
            (ix0, iy0, iz0),
            (ix0, iy0, iz1),
            (ix0, iy1, iz0),
            (ix0, iy1, iz1),
            (ix1, iy0, iz0),
            (ix1, iy0, iz1),
            (ix1, iy1, iz0),
            (ix1, iy1, iz1),
        ];
        let ws = [
            (1.0 - fx) * (1.0 - fy) * (1.0 - fz),
            (1.0 - fx) * (1.0 - fy) * fz,
            (1.0 - fx) * fy * (1.0 - fz),
            (1.0 - fx) * fy * fz,
            fx * (1.0 - fy) * (1.0 - fz),
            fx * (1.0 - fy) * fz,
            fx * fy * (1.0 - fz),
            fx * fy * fz,
        ];

        for c in 0..N_TRILINEAR {
            let (ci, cj, ck) = corners[c];
            all_indices[base + c] = ci * ny * nz + cj * nz + ck;
            all_weights[base + c] = ws[c];
        }
    }

    Ok((all_indices, all_weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::{tensor_log, DEFAULT_EPSILON};
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Vector3};

    /// Create a uniform field of identical tensors and verify interpolation
    /// returns the same tensor everywhere.
    #[test]
    fn test_uniform_field_interpolation() {
        let t = Matrix3::from_diagonal(&Vector3::new(2.0, 1.0, 0.5));
        let log_t = tensor_log(&t, DEFAULT_EPSILON).unwrap();

        // 4×4×4 grid, all the same tensor
        let grid_size = 64;
        let log_tensors: Vec<Matrix3<f64>> = vec![log_t; grid_size];

        // 2 FEM nodes, arbitrary weights (must sum to 1)
        let indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let weights = vec![
            0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15, // node 0
            0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, // node 1
        ];

        let result =
            interpolate_fem_points(&log_tensors, &indices, &weights, 2, grid_size).unwrap();

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], t, epsilon = 1e-10);
        assert_relative_eq!(result[1], t, epsilon = 1e-10);
    }

    /// Interpolation at a grid point should return exactly that tensor.
    #[test]
    fn test_interpolation_at_grid_point() {
        let t1 = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 1.0));
        let t2 = Matrix3::from_diagonal(&Vector3::new(3.0, 2.0, 1.0));

        let log_t1 = tensor_log(&t1, DEFAULT_EPSILON).unwrap();
        let log_t2 = tensor_log(&t2, DEFAULT_EPSILON).unwrap();

        let log_tensors = vec![log_t1, log_t2];

        // Node exactly at first grid point: weight 1.0 on index 0
        let indices = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let result = interpolate_fem_points(&log_tensors, &indices, &weights, 1, 2).unwrap();
        assert_relative_eq!(result[0], t1, epsilon = 1e-10);
    }

    /// Verify dimension mismatch detection.
    #[test]
    fn test_dimension_mismatch() {
        let log_tensors = vec![Matrix3::identity(); 8];
        let indices = vec![0usize; 4]; // Wrong: should be 8 for 1 node
        let weights = vec![0.125; 8];

        let result = interpolate_fem_points(&log_tensors, &indices, &weights, 1, 8);
        assert!(result.is_err());
    }

    /// Verify index bounds checking.
    #[test]
    fn test_index_out_of_bounds() {
        let log_tensors = vec![Matrix3::identity(); 4];
        let indices = vec![0, 1, 2, 3, 0, 1, 2, 999]; // 999 > grid_size
        let weights = vec![0.125; 8];

        let result = interpolate_fem_points(&log_tensors, &indices, &weights, 1, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_trilinear_weights_at_grid_center() {
        // Identity affine: voxel = physical coords
        let inv_affine = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let grid_shape = (4, 4, 4);

        // Point at exact center of a voxel cube: (0.5, 0.5, 0.5)
        let points = vec![0.5, 0.5, 0.5];
        let (indices, weights) =
            compute_trilinear_weights(&points, 1, &inv_affine, grid_shape).unwrap();

        // At the center, all 8 weights should be 1/8
        assert_eq!(indices.len(), 8);
        assert_eq!(weights.len(), 8);
        for w in &weights {
            assert_relative_eq!(*w, 0.125, epsilon = 1e-12);
        }

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_trilinear_weights_at_grid_vertex() {
        let inv_affine = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let grid_shape = (4, 4, 4);

        // Point at exact grid vertex (1, 1, 1)
        let points = vec![1.0, 1.0, 1.0];
        let (_, weights) = compute_trilinear_weights(&points, 1, &inv_affine, grid_shape).unwrap();

        // At a vertex, one weight should be 1.0 and rest 0
        let max_w = weights.iter().cloned().fold(0.0_f64, f64::max);
        assert_relative_eq!(max_w, 1.0, epsilon = 1e-12);

        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }
}
