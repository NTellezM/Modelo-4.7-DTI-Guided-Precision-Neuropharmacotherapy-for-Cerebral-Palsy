// src/validation.rs
//! Validation utilities for diffusion tensors.
//!
//! Provides batch checks for symmetry, positive definiteness, and statistical
//! summaries. Used both internally (after interpolation) and exposed to Python
//! for post-pipeline quality assurance.

use nalgebra::{Matrix3, SymmetricEigen};
use rayon::prelude::*;

use crate::tensor_ops::{fractional_anisotropy, symmetry_error};

/// Summary report for a batch of tensors.
#[derive(Debug, Clone)]
pub struct TensorValidationReport {
    pub n_total: usize,
    pub n_symmetric: usize,
    pub n_spd: usize,
    pub n_finite: usize,
    pub min_eigenvalue: f64,
    pub max_eigenvalue: f64,
    pub mean_eigenvalue: f64,
    pub mean_fa: f64,
    pub min_fa: f64,
    pub max_fa: f64,
    pub max_symmetry_error: f64,
}

impl TensorValidationReport {
    /// Check if all tensors pass basic quality requirements.
    pub fn all_valid(&self) -> bool {
        self.n_symmetric == self.n_total
            && self.n_spd == self.n_total
            && self.n_finite == self.n_total
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "TensorValidation: {n} tensors | SPD: {spd}/{n} | \
             eigenvalues: [{min_ev:.2e}, {max_ev:.2e}] (mean {mean_ev:.2e}) | \
             FA: [{min_fa:.3}, {max_fa:.3}] (mean {mean_fa:.3}) | \
             max symmetry error: {sym:.2e} | status: {status}",
            n = self.n_total,
            spd = self.n_spd,
            min_ev = self.min_eigenvalue,
            max_ev = self.max_eigenvalue,
            mean_ev = self.mean_eigenvalue,
            min_fa = self.min_fa,
            max_fa = self.max_fa,
            mean_fa = self.mean_fa,
            sym = self.max_symmetry_error,
            status = if self.all_valid() { "PASS" } else { "FAIL" },
        )
    }
}

/// Per-tensor validation result, computed in parallel.
struct SingleTensorCheck {
    is_finite: bool,
    is_symmetric: bool,
    is_spd: bool,
    min_eigenvalue: f64,
    max_eigenvalue: f64,
    mean_eigenvalue: f64,
    fa: f64,
    symmetry_error: f64,
}

/// Validate a batch of tensors in parallel.
///
/// Each tensor is provided as a row-major [f64; 9] in the flat array.
pub fn validate_tensors(
    tensors_flat: &[f64],
    n_tensors: usize,
    symmetry_tol: f64,
) -> TensorValidationReport {
    assert_eq!(tensors_flat.len(), n_tensors * 9);

    let checks: Vec<SingleTensorCheck> = (0..n_tensors)
        .into_par_iter()
        .map(|i| {
            let offset = i * 9;
            let t = Matrix3::from_row_slice(&tensors_flat[offset..offset + 9]);

            let is_finite = t.iter().all(|v| v.is_finite());
            let sym_err = symmetry_error(&t);
            let is_symmetric = sym_err < symmetry_tol;

            let eigen = SymmetricEigen::new(t);
            let evs = &eigen.eigenvalues;
            let min_ev = evs[0].min(evs[1]).min(evs[2]);
            let max_ev = evs[0].max(evs[1]).max(evs[2]);
            let mean_ev = (evs[0] + evs[1] + evs[2]) / 3.0;
            let is_spd = is_symmetric && min_ev > 0.0;

            let fa = if is_finite {
                fractional_anisotropy(&t)
            } else {
                f64::NAN
            };

            SingleTensorCheck {
                is_finite,
                is_symmetric,
                is_spd,
                min_eigenvalue: min_ev,
                max_eigenvalue: max_ev,
                mean_eigenvalue: mean_ev,
                fa,
                symmetry_error: sym_err,
            }
        })
        .collect();

    // Aggregate
    let mut report = TensorValidationReport {
        n_total: n_tensors,
        n_symmetric: 0,
        n_spd: 0,
        n_finite: 0,
        min_eigenvalue: f64::INFINITY,
        max_eigenvalue: f64::NEG_INFINITY,
        mean_eigenvalue: 0.0,
        mean_fa: 0.0,
        min_fa: f64::INFINITY,
        max_fa: f64::NEG_INFINITY,
        max_symmetry_error: 0.0,
    };

    for c in &checks {
        if c.is_finite {
            report.n_finite += 1;
        }
        if c.is_symmetric {
            report.n_symmetric += 1;
        }
        if c.is_spd {
            report.n_spd += 1;
        }
        if c.min_eigenvalue < report.min_eigenvalue {
            report.min_eigenvalue = c.min_eigenvalue;
        }
        if c.max_eigenvalue > report.max_eigenvalue {
            report.max_eigenvalue = c.max_eigenvalue;
        }
        report.mean_eigenvalue += c.mean_eigenvalue;
        if c.fa.is_finite() {
            report.mean_fa += c.fa;
            if c.fa < report.min_fa {
                report.min_fa = c.fa;
            }
            if c.fa > report.max_fa {
                report.max_fa = c.fa;
            }
        }
        if c.symmetry_error > report.max_symmetry_error {
            report.max_symmetry_error = c.symmetry_error;
        }
    }

    if n_tensors > 0 {
        report.mean_eigenvalue /= n_tensors as f64;
        report.mean_fa /= report.n_finite.max(1) as f64;
    }

    report
}

/// Quick check: are all tensors in the flat array finite?
pub fn all_finite(tensors_flat: &[f64]) -> bool {
    tensors_flat.par_iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::DEFAULT_SYMMETRY_TOL;
    use nalgebra::Vector3;

    #[test]
    fn test_validate_spd_tensors() {
        let t = Matrix3::from_diagonal(&Vector3::new(2.0, 1.0, 0.5));
        let mut flat = vec![0.0; 18];
        for i in 0..3 {
            for j in 0..3 {
                flat[i * 3 + j] = t[(i, j)];
                flat[9 + i * 3 + j] = t[(i, j)];
            }
        }

        let report = validate_tensors(&flat, 2, DEFAULT_SYMMETRY_TOL);
        assert!(report.all_valid());
        assert_eq!(report.n_spd, 2);
        assert!(report.min_eigenvalue > 0.0);
    }

    #[test]
    fn test_validate_non_spd_tensor() {
        // Tensor with a negative eigenvalue
        let t = Matrix3::from_diagonal(&Vector3::new(2.0, -0.5, 1.0));
        let mut flat = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                flat[i * 3 + j] = t[(i, j)];
            }
        }

        let report = validate_tensors(&flat, 1, DEFAULT_SYMMETRY_TOL);
        assert!(!report.all_valid());
        assert_eq!(report.n_spd, 0);
    }

    #[test]
    fn test_validate_nan_tensor() {
        let flat = vec![1.0, 0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0, 0.0, 1.0];
        let report = validate_tensors(&flat, 1, DEFAULT_SYMMETRY_TOL);
        assert!(!report.all_valid());
        assert_eq!(report.n_finite, 0);
    }

    #[test]
    fn test_all_finite() {
        assert!(all_finite(&[1.0, 2.0, 3.0]));
        assert!(!all_finite(&[1.0, f64::NAN, 3.0]));
        assert!(!all_finite(&[f64::INFINITY]));
    }
}
