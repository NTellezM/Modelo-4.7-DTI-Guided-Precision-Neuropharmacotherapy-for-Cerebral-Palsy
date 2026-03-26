#!/usr/bin/env python3
"""Generate synthetic DTI test data for CI and development.

Creates minimal test files that exercise all code paths without requiring
real medical imaging data.

Output:
    data/synthetic_dti.nii.gz   — 8×8×8 tensor volume (Voigt 6-component)
    data/synthetic_mesh.xdmf    — 200-node tetrahedral mesh
    data/synthetic_mesh.h5      — HDF5 backing for the XDMF mesh

Usage:
    python data/generate_synthetic.py
"""

from __future__ import annotations

from pathlib import Path

import meshio
import nibabel as nib
import numpy as np


def main() -> None:
    output_dir = Path(__file__).parent
    rng = np.random.default_rng(seed=2024)

    # --- DTI volume (8×8×8, Voigt format) ---
    shape = (8, 8, 8)
    dti = np.zeros((*shape, 6), dtype=np.float64)

    # Isotropic tensor: D = diag(1e-3, 1e-3, 1e-3)
    # Voigt: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
    dti[..., 0] = 1.0e-3  # Dxx
    dti[..., 3] = 1.0e-3  # Dyy
    dti[..., 5] = 1.0e-3  # Dzz

    # Add anisotropy along x in one quadrant
    dti[:4, :, :, 0] = 2.0e-3  # Higher Dxx in left half

    affine = np.eye(4, dtype=np.float64)
    img = nib.Nifti1Image(dti, affine)
    dti_path = output_dir / "synthetic_dti.nii.gz"
    nib.save(img, str(dti_path))
    print(f"Saved: {dti_path} ({dti_path.stat().st_size} bytes)")

    # --- Mesh (200 nodes inside [1, 6]^3) ---
    n_nodes = 200
    points = rng.uniform(1.0, 6.0, size=(n_nodes, 3))

    # Minimal tetrahedral connectivity (not a valid FEM mesh,
    # but sufficient for interpolation testing which only uses node coords)
    n_tet = 100
    simplices = np.column_stack([
        np.arange(n_tet),
        np.arange(1, n_tet + 1) % n_nodes,
        np.arange(2, n_tet + 2) % n_nodes,
        np.arange(3, n_tet + 3) % n_nodes,
    ])

    mesh = meshio.Mesh(points=points, cells=[("tetra", simplices)])
    mesh_path = output_dir / "synthetic_mesh.xdmf"
    meshio.write(str(mesh_path), mesh)
    print(f"Saved: {mesh_path}")

    # Verify files are loadable
    loaded_dti = nib.load(str(dti_path))
    assert loaded_dti.shape == (*shape, 6), f"Bad DTI shape: {loaded_dti.shape}"

    loaded_mesh = meshio.read(str(mesh_path))
    assert len(loaded_mesh.points) == n_nodes, f"Bad node count: {len(loaded_mesh.points)}"

    print(f"\nSynthetic data ready:")
    print(f"  DTI: {shape} volume, {np.prod(shape)} voxels")
    print(f"  Mesh: {n_nodes} nodes, {n_tet} tetrahedra")


if __name__ == "__main__":
    main()
