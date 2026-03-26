"""High-level pipeline: DTI volume → interpolated FEM tensors in one call.

This module ties together I/O, Rust computation, and validation into a single
function that can be called programmatically or from the CLI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MappingResult:
    """Result of the tensor mapping pipeline."""

    n_voxels: int
    n_nodes: int
    output_path: str
    elapsed_precompute_s: float
    elapsed_weights_s: float
    elapsed_interpolation_s: float
    elapsed_total_s: float
    validation: dict[str, Any] | None = None

    def summary(self) -> str:
        lines = [
            "DTI Mapping complete:",
            f"  Voxels: {self.n_voxels:,}",
            f"  FEM nodes: {self.n_nodes:,}",
            f"  Precompute logs: {self.elapsed_precompute_s:.2f}s",
            f"  Trilinear weights: {self.elapsed_weights_s:.2f}s",
            f"  Interpolation: {self.elapsed_interpolation_s:.2f}s",
            f"  Total: {self.elapsed_total_s:.2f}s",
            f"  Output: {self.output_path}",
        ]
        if self.validation:
            status = "PASS" if self.validation.get("all_valid", False) else "FAIL"
            lines.append(f"  Validation: {status}")
        return "\n".join(lines)


def map_tensors_to_mesh(
    dti_path: str | Path,
    mesh_path: str | Path,
    output_path: str | Path,
    *,
    epsilon: float = 1e-12,
    validate: bool = True,
    output_format: str = "h5",
) -> MappingResult:
    """Map DTI tensors from a voxel grid to FEM mesh nodes.

    This is the main entry point for the dti_mapper pipeline.

    Parameters
    ----------
    dti_path : str or Path
        Path to DTI NIfTI file (.nii or .nii.gz).
    mesh_path : str or Path
        Path to mesh XDMF file.
    output_path : str or Path
        Where to save the interpolated tensors.
    epsilon : float
        Clamping threshold for small eigenvalues in matrix log.
    validate : bool
        Whether to validate the output tensors.
    output_format : str
        "h5" for HDF5 with metadata, "npy" for plain NumPy.

    Returns
    -------
    MappingResult
        Timing and validation information.
    """
    from dti_mapper._rust import (
        compute_trilinear_weights_py,
        interpolate_fem_points_py,
        precompute_logs,
        validate_tensors_py,
    )
    from dti_mapper.io import (
        compute_inverse_affine,
        load_dti_nifti,
        load_mesh_nodes,
        save_tensors_h5,
        save_tensors_npy,
    )

    t_start = time.perf_counter()

    # --- Step 1: Load data ---
    logger.info("Step 1/4: Loading data")
    tensors_vol, affine = load_dti_nifti(dti_path)
    nodes = load_mesh_nodes(mesh_path)

    spatial_shape = tensors_vol.shape[:3]
    n_voxels = int(np.prod(spatial_shape))
    n_nodes = len(nodes)

    logger.info("  DTI volume: %s (%d voxels)", spatial_shape, n_voxels)
    logger.info("  FEM mesh: %d nodes", n_nodes)

    # Flatten volume to (N_voxels, 3, 3)
    tensors_flat = tensors_vol.reshape(-1, 3, 3)

    # --- Step 2: Precompute logarithms ---
    logger.info("Step 2/4: Precomputing matrix logarithms")
    t2 = time.perf_counter()

    log_tensors = precompute_logs(
        np.ascontiguousarray(tensors_flat, dtype=np.float64),
        epsilon,
    )

    elapsed_precompute = time.perf_counter() - t2
    logger.info("  Done in %.2fs (%.0f tensors/s)", elapsed_precompute,
                n_voxels / max(elapsed_precompute, 1e-6))

    # --- Step 3: Compute trilinear weights ---
    logger.info("Step 3/4: Computing trilinear weights")
    t3 = time.perf_counter()

    inv_affine = compute_inverse_affine(affine)
    indices, weights = compute_trilinear_weights_py(
        np.ascontiguousarray(nodes, dtype=np.float64),
        inv_affine,
        spatial_shape,
    )

    elapsed_weights = time.perf_counter() - t3
    logger.info("  Done in %.2fs", elapsed_weights)

    # --- Step 4: Interpolate ---
    logger.info("Step 4/4: Interpolating tensors at FEM nodes")
    t4 = time.perf_counter()

    result_flat = interpolate_fem_points_py(log_tensors, indices, weights)
    # Reshape from (N, 9) to (N, 3, 3)
    result = result_flat.reshape(-1, 3, 3)

    elapsed_interp = time.perf_counter() - t4
    logger.info("  Done in %.2fs (%.0f nodes/s)", elapsed_interp,
                n_nodes / max(elapsed_interp, 1e-6))

    # --- Save ---
    output_path = Path(output_path)
    if output_format == "h5":
        save_tensors_h5(
            result,
            output_path,
            mesh_path=mesh_path,
            dti_path=dti_path,
            metadata={"epsilon": epsilon, "grid_shape": list(spatial_shape)},
        )
    else:
        save_tensors_npy(result, output_path)

    elapsed_total = time.perf_counter() - t_start

    # --- Validate ---
    validation_report = None
    if validate:
        logger.info("Validating output tensors")
        val = validate_tensors_py(
            np.ascontiguousarray(result, dtype=np.float64),
        )
        validation_report = dict(val)
        if val["all_valid"]:
            logger.info("Validation PASSED: all %d tensors are SPD", n_nodes)
        else:
            logger.warning("Validation FAILED: %s", val["summary"])

    mapping_result = MappingResult(
        n_voxels=n_voxels,
        n_nodes=n_nodes,
        output_path=str(output_path),
        elapsed_precompute_s=elapsed_precompute,
        elapsed_weights_s=elapsed_weights,
        elapsed_interpolation_s=elapsed_interp,
        elapsed_total_s=elapsed_total,
        validation=validation_report,
    )

    logger.info("\n%s", mapping_result.summary())
    return mapping_result
