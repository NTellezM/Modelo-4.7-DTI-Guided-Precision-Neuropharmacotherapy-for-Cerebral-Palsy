"""I/O utilities for loading DTI volumes, mesh nodes, and saving interpolated tensors.

This module handles the translation between file formats (NIfTI, XDMF/HDF5, NumPy)
and the internal array representations expected by the Rust interpolation engine.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import meshio
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# Standard ordering of the 6 unique components of a 3×3 symmetric tensor:
# (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
_VOIGT_INDICES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def load_dti_nifti(
    path: str | Path,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a DTI tensor volume from a NIfTI file.

    Supports two common formats:
    - 4D volume (Nx, Ny, Nz, 6): 6 unique symmetric components in Voigt order.
    - 5D volume (Nx, Ny, Nz, 3, 3): full 3×3 tensor at each voxel.

    Parameters
    ----------
    path : str or Path
        Path to the .nii or .nii.gz file.
    expected_shape : tuple, optional
        If provided, verify the spatial dimensions match.

    Returns
    -------
    tensors : np.ndarray, shape (Nx, Ny, Nz, 3, 3), dtype float64
        Full 3×3 symmetric tensors at each voxel.
    affine : np.ndarray, shape (4, 4), dtype float64
        Affine transformation matrix (voxel → physical RAS coordinates).

    Raises
    ------
    ValueError
        If the file format is unrecognized or dimensions don't match.
    """
    path = Path(path)
    logger.info("Loading DTI volume from %s", path)

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    affine = np.asarray(img.affine, dtype=np.float64)

    if data.ndim == 4 and data.shape[3] == 6:
        # Voigt notation → full 3×3
        logger.info("Converting from 6-component Voigt format")
        spatial_shape = data.shape[:3]
        tensors = np.zeros((*spatial_shape, 3, 3), dtype=np.float64)

        for idx, (i, j) in enumerate(_VOIGT_INDICES):
            tensors[..., i, j] = data[..., idx]
            if i != j:
                tensors[..., j, i] = data[..., idx]  # Symmetry

    elif data.ndim == 5 and data.shape[3:] == (3, 3):
        tensors = data
    else:
        msg = (
            f"Unsupported DTI volume shape {data.shape}. "
            "Expected (Nx, Ny, Nz, 6) or (Nx, Ny, Nz, 3, 3)."
        )
        raise ValueError(msg)

    spatial_shape = tensors.shape[:3]
    n_voxels = np.prod(spatial_shape)
    logger.info(
        "Loaded DTI volume: shape %s, %d voxels, affine det=%.4f",
        spatial_shape,
        n_voxels,
        np.linalg.det(affine[:3, :3]),
    )

    if expected_shape is not None and spatial_shape != expected_shape:
        msg = f"Spatial shape mismatch: expected {expected_shape}, got {spatial_shape}"
        raise ValueError(msg)

    # Quick sanity check: any NaN?
    n_nan = np.count_nonzero(np.isnan(tensors))
    if n_nan > 0:
        logger.warning("DTI volume contains %d NaN values (%.2f%%)", n_nan, 100 * n_nan / tensors.size)

    return tensors, affine


def load_mesh_nodes(
    xdmf_path: str | Path,
) -> np.ndarray:
    """Read FEM mesh node coordinates from an XDMF file.

    Parameters
    ----------
    xdmf_path : str or Path
        Path to the mesh.xdmf file.

    Returns
    -------
    np.ndarray, shape (N_nodes, 3), dtype float64
        Physical coordinates (x, y, z) of each mesh node.
    """
    path = Path(xdmf_path)
    logger.info("Loading mesh nodes from %s", path)

    mesh = meshio.read(str(path))
    points = np.asarray(mesh.points, dtype=np.float64)

    if points.shape[1] == 2:
        logger.info("2D mesh detected, padding z-coordinate with zeros")
        points = np.column_stack([points, np.zeros(len(points))])

    logger.info("Loaded %d mesh nodes, bounding box: [%.2f, %.2f] × [%.2f, %.2f] × [%.2f, %.2f]",
                len(points),
                points[:, 0].min(), points[:, 0].max(),
                points[:, 1].min(), points[:, 1].max(),
                points[:, 2].min(), points[:, 2].max())

    return points


def compute_inverse_affine(affine: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 4×4 affine matrix.

    Parameters
    ----------
    affine : np.ndarray, shape (4, 4)
        Forward affine (voxel → physical).

    Returns
    -------
    np.ndarray, shape (4, 4)
        Inverse affine (physical → voxel).

    Raises
    ------
    ValueError
        If the affine is singular.
    """
    det = np.linalg.det(affine)
    if abs(det) < 1e-12:
        msg = f"Affine matrix is singular (det = {det:.2e})"
        raise ValueError(msg)
    return np.linalg.inv(affine).astype(np.float64)


def _compute_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_tensors_h5(
    tensors: np.ndarray,
    path: str | Path,
    *,
    mesh_path: str | Path | None = None,
    dti_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save interpolated tensors to HDF5 with metadata for traceability.

    The file contains:
    - Dataset "tensors": shape (N_nodes, 3, 3), float64
    - Attributes: timestamps, file hashes, custom metadata

    Parameters
    ----------
    tensors : np.ndarray, shape (N_nodes, 3, 3)
        Interpolated SPD tensors.
    path : str or Path
        Output file path (.h5 or .hdf5).
    mesh_path : str or Path, optional
        Path to the source mesh, for hash recording.
    dti_path : str or Path, optional
        Path to the source DTI volume, for hash recording.
    metadata : dict, optional
        Additional metadata to store as HDF5 attributes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving %d tensors to %s", len(tensors), path)

    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "tensors",
            data=tensors,
            dtype="float64",
            compression="gzip",
            compression_opts=4,
            chunks=(min(1000, len(tensors)), 3, 3),
        )

        # Standard metadata
        ds.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        ds.attrs["dti_mapper_version"] = "0.1.0"
        ds.attrs["n_nodes"] = len(tensors)

        if mesh_path is not None:
            mesh_path = Path(mesh_path)
            if mesh_path.exists():
                ds.attrs["mesh_sha256"] = _compute_sha256(mesh_path)
                ds.attrs["mesh_file"] = str(mesh_path)

        if dti_path is not None:
            dti_path = Path(dti_path)
            if dti_path.exists():
                ds.attrs["dti_sha256"] = _compute_sha256(dti_path)
                ds.attrs["dti_file"] = str(dti_path)

        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v

    logger.info("Saved successfully: %.2f MB", path.stat().st_size / 1e6)


def save_tensors_npy(
    tensors: np.ndarray,
    path: str | Path,
) -> None:
    """Save interpolated tensors as a NumPy .npy file (no metadata).

    Parameters
    ----------
    tensors : np.ndarray, shape (N_nodes, 3, 3)
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), tensors)
    logger.info("Saved %d tensors to %s (%.2f MB)", len(tensors), path, path.stat().st_size / 1e6)


def load_tensors_h5(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load tensors and metadata from an HDF5 file.

    Returns
    -------
    tensors : np.ndarray, shape (N_nodes, 3, 3)
    metadata : dict
        All HDF5 attributes.
    """
    path = Path(path)
    with h5py.File(str(path), "r") as f:
        ds = f["tensors"]
        tensors = np.asarray(ds, dtype=np.float64)
        metadata = dict(ds.attrs)
    return tensors, metadata
