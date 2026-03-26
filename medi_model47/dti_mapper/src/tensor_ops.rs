// src/tensor_ops.rs
//! Matrix logarithm and exponential for 3×3 symmetric positive definite tensors.
//!
//! Uses spectral decomposition via `nalgebra::SymmetricEigen`. All operations
//! preserve symmetry by construction (V * diag(f(λ)) * V^T).

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use crate::errors::{DtiError, DtiResult};

/// Default clamping threshold to avoid log(0).
pub const DEFAULT_EPSILON: f64 = 1e-12;

/// Tolerance for symmetry check: ||T - T^T||_F.
pub const DEFAULT_SYMMETRY_TOL: f64 = 1e-8;

/// Compute the matrix logarithm of a 3×3 symmetric positive definite tensor.
///
/// Given T = V * diag(λ₁, λ₂, λ₃) * V^T, returns V * diag(ln λ₁, ln λ₂, ln λ₃) * V^T.
///
/// Eigenvalues below `epsilon` are clamped to `epsilon` to ensure numerical stability.
/// This is the standard approach in log-Euclidean DTI processing.
///
/// # Errors
///
/// Returns `DtiError::NonFinite` if the input contains NaN or Inf.
pub fn tensor_log(t: &Matrix3<f64>, epsilon: f64) -> DtiResult<Matrix3<f64>> {
    // Check for non-finite values
    if !t.iter().all(|v| v.is_finite()) {
        return Err(DtiError::NonFinite { position: 0 });
    }

    let eigen = SymmetricEigen::new(*t);

    let log_values = Vector3::new(
        eigen.eigenvalues[0].max(epsilon).ln(),
        eigen.eigenvalues[1].max(epsilon).ln(),
        eigen.eigenvalues[2].max(epsilon).ln(),
    );

    let v = &eigen.eigenvectors;
    Ok(v * Matrix3::from_diagonal(&log_values) * v.transpose())
}

/// Compute the matrix exponential of a 3×3 symmetric tensor.
///
/// Given T = V * diag(λ₁, λ₂, λ₃) * V^T, returns V * diag(e^λ₁, e^λ₂, e^λ₃) * V^T.
///
/// The result is always symmetric positive definite since exp(λ) > 0.
///
/// # Errors
///
/// Returns `DtiError::NonFinite` if the input contains NaN or Inf.
pub fn tensor_exp(t: &Matrix3<f64>) -> DtiResult<Matrix3<f64>> {
    if !t.iter().all(|v| v.is_finite()) {
        return Err(DtiError::NonFinite { position: 0 });
    }

    let eigen = SymmetricEigen::new(*t);

    let exp_values = Vector3::new(
        eigen.eigenvalues[0].exp(),
        eigen.eigenvalues[1].exp(),
        eigen.eigenvalues[2].exp(),
    );

    let v = &eigen.eigenvectors;
    Ok(v * Matrix3::from_diagonal(&exp_values) * v.transpose())
}

/// Compute the Frobenius norm of the difference between a tensor and its transpose.
///
/// A perfectly symmetric tensor returns 0.0.
#[inline]
pub fn symmetry_error(t: &Matrix3<f64>) -> f64 {
    (t - t.transpose()).norm()
}

/// Compute fractional anisotropy (FA) of a diffusion tensor.
///
/// FA = sqrt(3/2) * ||T - (trace(T)/3)*I|| / ||T||
///
/// Returns a value in [0, 1]: 0 = perfectly isotropic, 1 = perfectly anisotropic.
pub fn fractional_anisotropy(t: &Matrix3<f64>) -> f64 {
    let eigen = SymmetricEigen::new(*t);
    let e = &eigen.eigenvalues;
    let mean = (e[0] + e[1] + e[2]) / 3.0;

    let numerator = ((e[0] - mean).powi(2) + (e[1] - mean).powi(2) + (e[2] - mean).powi(2)).sqrt();
    let denominator = (e[0].powi(2) + e[1].powi(2) + e[2].powi(2)).sqrt();

    if denominator < 1e-30 {
        return 0.0;
    }

    (3.0_f64 / 2.0).sqrt() * numerator / denominator
}

/// Compute the weighted log-Euclidean mean of a set of SPD tensors.
///
/// Given tensors T₁..Tₙ and weights w₁..wₙ (summing to 1),
/// the log-Euclidean mean is: exp(Σ wᵢ * log(Tᵢ))
///
/// This is the core operation used in interpolation.
pub fn log_euclidean_mean(
    tensors: &[Matrix3<f64>],
    weights: &[f64],
    epsilon: f64,
) -> DtiResult<Matrix3<f64>> {
    debug_assert_eq!(tensors.len(), weights.len());

    let mut sum = Matrix3::zeros();
    for (t, &w) in tensors.iter().zip(weights.iter()) {
        if w.abs() < 1e-15 {
            continue;
        }
        let log_t = tensor_log(t, epsilon)?;
        sum += log_t * w;
    }

    tensor_exp(&sum)
}

/// Precompute matrix logarithms for an entire volume of tensors.
///
/// Input: flat slice of N tensors, each as a row-major [f64; 9].
/// Output: Vec of N Matrix3<f64> logarithms.
///
/// Uses rayon for parallel computation.
pub fn precompute_logs_batch(
    tensors_flat: &[f64],
    n_tensors: usize,
    epsilon: f64,
) -> DtiResult<Vec<Matrix3<f64>>> {
    use rayon::prelude::*;

    if tensors_flat.len() != n_tensors * 9 {
        return Err(DtiError::DimensionMismatch {
            expected: format!("{}", n_tensors * 9),
            got: format!("{}", tensors_flat.len()),
            context: "precompute_logs_batch: flat tensor array".into(),
        });
    }

    let results: Vec<DtiResult<Matrix3<f64>>> = (0..n_tensors)
        .into_par_iter()
        .map(|i| {
            let offset = i * 9;
            let t = Matrix3::from_row_slice(&tensors_flat[offset..offset + 9]);
            tensor_log(&t, epsilon)
        })
        .collect();

    // Collect results, propagating the first error
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Matrix3;

    /// Helper: create a random SPD tensor via A * A^T + εI.
    fn random_spd(seed: u64) -> Matrix3<f64> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let a = Matrix3::from_fn(|_, _| normal.sample(&mut rng));
        let spd = a * a.transpose() + Matrix3::identity() * 0.01;
        spd
    }

    #[test]
    fn test_log_exp_roundtrip() {
        for seed in 0..100 {
            let t = random_spd(seed);
            let log_t = tensor_log(&t, DEFAULT_EPSILON).unwrap();
            let recovered = tensor_exp(&log_t).unwrap();
            assert_relative_eq!(t, recovered, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_log_identity_is_zero() {
        let id = Matrix3::identity();
        let log_id = tensor_log(&id, DEFAULT_EPSILON).unwrap();
        assert_relative_eq!(log_id, Matrix3::zeros(), epsilon = 1e-12);
    }

    #[test]
    fn test_exp_zero_is_identity() {
        let zero = Matrix3::zeros();
        let exp_zero = tensor_exp(&zero).unwrap();
        assert_relative_eq!(exp_zero, Matrix3::identity(), epsilon = 1e-12);
    }

    #[test]
    fn test_log_preserves_symmetry() {
        let t = random_spd(42);
        let log_t = tensor_log(&t, DEFAULT_EPSILON).unwrap();
        assert!(symmetry_error(&log_t) < 1e-14);
    }

    #[test]
    fn test_exp_result_is_spd() {
        // Any symmetric matrix should yield an SPD tensor under exp
        let sym = Matrix3::new(1.0, 0.5, 0.2, 0.5, 2.0, 0.3, 0.2, 0.3, 1.5);
        let exp_t = tensor_exp(&sym).unwrap();
        let eigen = SymmetricEigen::new(exp_t);
        assert!(eigen.eigenvalues.min() > 0.0);
    }

    #[test]
    fn test_fractional_anisotropy_isotropic() {
        let iso = Matrix3::identity() * 1.5;
        let fa = fractional_anisotropy(&iso);
        assert!(fa.abs() < 1e-12, "Isotropic tensor should have FA ≈ 0");
    }

    #[test]
    fn test_fractional_anisotropy_anisotropic() {
        let aniso = Matrix3::from_diagonal(&Vector3::new(3.0, 0.1, 0.1));
        let fa = fractional_anisotropy(&aniso);
        assert!(
            fa > 0.9,
            "Highly anisotropic tensor should have FA close to 1"
        );
    }

    #[test]
    fn test_log_euclidean_mean_equal_weights() {
        let t1 = random_spd(1);
        let t2 = random_spd(2);
        let mean = log_euclidean_mean(&[t1, t2], &[0.5, 0.5], DEFAULT_EPSILON).unwrap();

        // Mean should be SPD
        let eigen = SymmetricEigen::new(mean);
        assert!(eigen.eigenvalues.min() > 0.0);

        // Mean should be symmetric
        assert!(symmetry_error(&mean) < 1e-14);
    }

    #[test]
    fn test_log_euclidean_mean_single_tensor() {
        let t = random_spd(7);
        let mean = log_euclidean_mean(&[t], &[1.0], DEFAULT_EPSILON).unwrap();
        assert_relative_eq!(t, mean, epsilon = 1e-10);
    }

    #[test]
    fn test_precompute_logs_batch() {
        let t1 = random_spd(10);
        let t2 = random_spd(11);
        let mut flat = vec![0.0; 18];
        for i in 0..3 {
            for j in 0..3 {
                flat[i * 3 + j] = t1[(i, j)];
                flat[9 + i * 3 + j] = t2[(i, j)];
            }
        }

        let logs = precompute_logs_batch(&flat, 2, DEFAULT_EPSILON).unwrap();
        assert_eq!(logs.len(), 2);

        // Verify against individual log
        let log_t1 = tensor_log(&t1, DEFAULT_EPSILON).unwrap();
        assert_relative_eq!(logs[0], log_t1, epsilon = 1e-12);
    }

    #[test]
    fn test_nan_input_error() {
        let bad = Matrix3::new(1.0, 0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0, 0.0, 1.0);
        assert!(tensor_log(&bad, DEFAULT_EPSILON).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let flat = vec![0.0; 10]; // Wrong size
        let result = precompute_logs_batch(&flat, 2, DEFAULT_EPSILON);
        assert!(result.is_err());
    }
}
