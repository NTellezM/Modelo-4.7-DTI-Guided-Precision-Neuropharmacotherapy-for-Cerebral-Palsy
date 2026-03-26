"""Validation utilities for interpolated tensor outputs.

Provides high-level validation that checks data contract compliance:
tensor dimensions match mesh nodes, all tensors are SPD, and metadata
is consistent with source files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TensorValidationReport:
    """Structured report from tensor validation."""

    n_tensors: int = 0
    n_mesh_nodes: int = 0
    shape_match: bool = False
    n_finite: int = 0
    n_symmetric: int = 0
    n_spd: int = 0
    min_eigenvalue: float = float("inf")
    max_eigenvalue: float = float("-inf")
    mean_fa: float = 0.0
    mesh_hash_match: bool | None = None  # None if not checked
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Tensor validation: {status}",
            f"  Tensors: {self.n_tensors}, Mesh nodes: {self.n_mesh_nodes}",
            f"  Shape match: {self.shape_match}",
            f"  Finite: {self.n_finite}/{self.n_tensors}",
            f"  Symmetric: {self.n_symmetric}/{self.n_tensors}",
            f"  SPD: {self.n_spd}/{self.n_tensors}",
            f"  Eigenvalues: [{self.min_eigenvalue:.2e}, {self.max_eigenvalue:.2e}]",
            f"  Mean FA: {self.mean_fa:.4f}",
        ]
        if self.mesh_hash_match is not None:
            lines.append(f"  Mesh hash match: {self.mesh_hash_match}")
        if self.errors:
            lines.append(f"  Errors: {'; '.join(self.errors)}")
        if self.warnings:
            lines.append(f"  Warnings: {'; '.join(self.warnings)}")
        return "\n".join(lines)


def validate_tensor_output(
    tensor_path: str | Path,
    mesh_path: str | Path,
    *,
    spd_tolerance: float = 1e-10,
    symmetry_tolerance: float = 1e-8,
    check_mesh_hash: bool = True,
) -> TensorValidationReport:
    """Validate that interpolated tensors comply with the data contract.

    Checks performed:
    1. First dimension matches number of mesh nodes.
    2. All values are finite (no NaN or Inf).
    3. All tensors are symmetric within tolerance.
    4. All tensors are positive definite (min eigenvalue > 0).
    5. Optionally verify mesh hash matches stored metadata.

    Parameters
    ----------
    tensor_path : str or Path
        Path to tensors file (.h5 or .npy).
    mesh_path : str or Path
        Path to mesh XDMF file.
    spd_tolerance : float
        Minimum acceptable eigenvalue for positive definiteness.
    symmetry_tolerance : float
        Maximum Frobenius norm of (T - T^T) for symmetry.
    check_mesh_hash : bool
        Whether to verify mesh file hash against stored metadata.

    Returns
    -------
    TensorValidationReport
    """
    report = TensorValidationReport()

    tensor_path = Path(tensor_path)
    mesh_path = Path(mesh_path)

    # Load tensors
    try:
        if tensor_path.suffix in (".h5", ".hdf5"):
            from dti_mapper.io import load_tensors_h5
            tensors, metadata = load_tensors_h5(tensor_path)
        elif tensor_path.suffix == ".npy":
            tensors = np.load(str(tensor_path))
            metadata = {}
        else:
            report.errors.append(f"Unsupported tensor format: {tensor_path.suffix}")
            return report
    except Exception as e:
        report.errors.append(f"Failed to load tensors: {e}")
        return report

    # Load mesh node count
    try:
        from dti_mapper.io import load_mesh_nodes
        nodes = load_mesh_nodes(mesh_path)
        report.n_mesh_nodes = len(nodes)
    except Exception as e:
        report.errors.append(f"Failed to load mesh: {e}")
        return report

    # Reshape if needed
    if tensors.ndim == 2 and tensors.shape[1] == 9:
        tensors = tensors.reshape(-1, 3, 3)

    report.n_tensors = len(tensors)

    # Check 1: Shape match
    report.shape_match = report.n_tensors == report.n_mesh_nodes
    if not report.shape_match:
        report.errors.append(
            f"Dimension mismatch: {report.n_tensors} tensors vs "
            f"{report.n_mesh_nodes} mesh nodes"
        )

    # Check 2: Finiteness
    is_finite = np.isfinite(tensors).all(axis=(1, 2))
    report.n_finite = int(is_finite.sum())
    if report.n_finite < report.n_tensors:
        n_bad = report.n_tensors - report.n_finite
        report.errors.append(f"{n_bad} tensors contain NaN or Inf")

    # Check 3 & 4: Symmetry and positive definiteness
    # Use Rust validation if available for performance
    try:
        from dti_mapper._rust import validate_tensors_py
        if tensors.ndim == 3:
            rust_report = validate_tensors_py(
                np.ascontiguousarray(tensors, dtype=np.float64),
                symmetry_tolerance,
            )
            report.n_symmetric = rust_report["n_symmetric"]
            report.n_spd = rust_report["n_spd"]
            report.min_eigenvalue = rust_report["min_eigenvalue"]
            report.max_eigenvalue = rust_report["max_eigenvalue"]
            report.mean_fa = rust_report["mean_fa"]
        else:
            report.warnings.append("Tensors not in (N, 3, 3) shape, skipping Rust validation")
            _validate_python_fallback(tensors, report, spd_tolerance, symmetry_tolerance)
    except ImportError:
        logger.warning("Rust extension not available, using Python fallback")
        _validate_python_fallback(tensors, report, spd_tolerance, symmetry_tolerance)

    if report.n_spd < report.n_tensors:
        n_bad = report.n_tensors - report.n_spd
        report.errors.append(f"{n_bad} tensors are not positive definite")

    if report.n_symmetric < report.n_tensors:
        n_bad = report.n_tensors - report.n_symmetric
        report.errors.append(f"{n_bad} tensors are not symmetric")

    # Check 5: Mesh hash
    if check_mesh_hash and "mesh_sha256" in metadata:
        from dti_mapper.io import _compute_sha256
        # Find the .h5 file associated with the mesh
        h5_path = mesh_path.parent / mesh_path.with_suffix(".h5").name
        if h5_path.exists():
            actual_hash = _compute_sha256(h5_path)
            report.mesh_hash_match = actual_hash == metadata["mesh_sha256"]
            if not report.mesh_hash_match:
                report.warnings.append(
                    "Mesh hash does not match stored metadata — tensors may have "
                    "been computed with a different mesh"
                )
        else:
            report.mesh_hash_match = None
            report.warnings.append(f"Mesh HDF5 file not found at {h5_path}")

    # Overall pass/fail
    report.passed = len(report.errors) == 0

    logger.info("\n%s", report.summary())
    return report


def _validate_python_fallback(
    tensors: np.ndarray,
    report: TensorValidationReport,
    spd_tol: float,
    sym_tol: float,
) -> None:
    """Pure Python/NumPy validation fallback (slower but no Rust dependency)."""
    n = len(tensors)

    # Symmetry
    sym_errors = np.linalg.norm(tensors - tensors.transpose(0, 2, 1), axis=(1, 2))
    report.n_symmetric = int((sym_errors < sym_tol).sum())

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(tensors)  # (N, 3), sorted ascending
    report.min_eigenvalue = float(eigenvalues.min())
    report.max_eigenvalue = float(eigenvalues.max())
    report.n_spd = int((eigenvalues[:, 0] > spd_tol).sum())

    # FA
    mean_ev = eigenvalues.mean(axis=1, keepdims=True)
    numerator = np.sqrt(((eigenvalues - mean_ev) ** 2).sum(axis=1))
    denominator = np.sqrt((eigenvalues ** 2).sum(axis=1))
    fa = np.where(denominator > 1e-30, np.sqrt(1.5) * numerator / denominator, 0.0)
    report.mean_fa = float(fa.mean())
