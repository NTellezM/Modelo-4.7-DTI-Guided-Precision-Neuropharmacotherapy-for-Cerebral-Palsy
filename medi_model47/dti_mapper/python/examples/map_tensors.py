#!/usr/bin/env python3
"""Example: Map DTI tensors to FEM mesh nodes.

This script demonstrates the complete dti_mapper pipeline:
1. Load a DTI volume (NIfTI) and a FEM mesh (XDMF).
2. Precompute matrix logarithms of all voxel tensors.
3. Compute trilinear interpolation weights for each mesh node.
4. Interpolate tensors in log-Euclidean space.
5. Validate and save the result.

Usage:
    python map_tensors.py --dti data/dti.nii.gz --mesh mesh/mesh.xdmf --output tensors.h5
    python map_tensors.py --synthetic --output tensors.h5  # Use synthetic data

Requirements:
    pip install dti_mapper nibabel meshio h5py numpy
    (Rust extension must be compiled: maturin develop --release)
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("map_tensors")


def create_synthetic_data(
    output_dir: Path,
    grid_shape: tuple[int, int, int] = (32, 32, 32),
    n_mesh_nodes: int = 5000,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate synthetic DTI volume and mesh for testing.

    Creates:
    - A NIfTI file with anisotropic tensors (principal direction varies spatially).
    - An XDMF mesh file with random nodes inside the grid.

    Returns (dti_path, mesh_path).
    """
    import nibabel as nib
    import meshio

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = grid_shape
    voxel_size = 1.0  # mm

    # --- DTI volume ---
    # Create tensors with a helical fiber pattern:
    # Principal direction rotates around the z-axis as z increases
    logger.info("Generating synthetic DTI volume (%d × %d × %d)", nx, ny, nz)

    dti_6comp = np.zeros((nx, ny, nz, 6), dtype=np.float64)

    for iz in range(nz):
        theta = 2 * np.pi * iz / nz  # Rotation angle

        # Eigenvalues: anisotropic (axial >> radial)
        lambda_axial = 1.7e-3   # mm²/s (typical white matter)
        lambda_radial = 0.3e-3  # mm²/s

        # Principal direction rotates in xy-plane
        e1 = np.array([np.cos(theta), np.sin(theta), 0.0])
        e2 = np.array([-np.sin(theta), np.cos(theta), 0.0])
        e3 = np.array([0.0, 0.0, 1.0])

        # Reconstruct tensor: D = λ₁·e₁⊗e₁ + λ₂·e₂⊗e₂ + λ₃·e₃⊗e₃
        D = (lambda_axial * np.outer(e1, e1)
             + lambda_radial * np.outer(e2, e2)
             + lambda_radial * np.outer(e3, e3))

        # Store in Voigt order: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
        dti_6comp[:, :, iz, 0] = D[0, 0]
        dti_6comp[:, :, iz, 1] = D[0, 1]
        dti_6comp[:, :, iz, 2] = D[0, 2]
        dti_6comp[:, :, iz, 3] = D[1, 1]
        dti_6comp[:, :, iz, 4] = D[1, 2]
        dti_6comp[:, :, iz, 5] = D[2, 2]

    # Add small noise to make it realistic
    noise_scale = 1e-5
    dti_6comp += rng.normal(0, noise_scale, dti_6comp.shape)

    # Ensure SPD by adjusting diagonal
    for comp_idx in [0, 3, 5]:  # Dxx, Dyy, Dzz
        dti_6comp[..., comp_idx] = np.maximum(dti_6comp[..., comp_idx], 1e-6)

    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    dti_path = output_dir / "synthetic_dti.nii.gz"
    nib.save(nib.Nifti1Image(dti_6comp, affine), str(dti_path))
    logger.info("  Saved DTI: %s (%.2f MB)", dti_path, dti_path.stat().st_size / 1e6)

    # --- Mesh ---
    logger.info("Generating synthetic mesh (%d nodes)", n_mesh_nodes)

    # Random points inside the grid (with margin to avoid boundary effects)
    margin = 2.0
    points = rng.uniform(
        margin, min(nx, ny, nz) * voxel_size - margin,
        size=(n_mesh_nodes, 3),
    )

    # Create a simple tetrahedral mesh using Delaunay triangulation
    # For this example, we just need the points — the interpolation
    # only uses node coordinates, not connectivity
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        cells = [("tetra", tri.simplices)]
    except ImportError:
        # Fallback: create dummy tetrahedra
        logger.warning("scipy not available, creating minimal connectivity")
        n_tet = min(n_mesh_nodes - 4, 1000)
        simplices = np.column_stack([
            rng.integers(0, n_mesh_nodes, n_tet),
            rng.integers(0, n_mesh_nodes, n_tet),
            rng.integers(0, n_mesh_nodes, n_tet),
            rng.integers(0, n_mesh_nodes, n_tet),
        ])
        cells = [("tetra", simplices)]

    mesh_path = output_dir / "synthetic_mesh.xdmf"
    mesh = meshio.Mesh(points=points, cells=cells)
    meshio.write(str(mesh_path), mesh)
    logger.info("  Saved mesh: %s", mesh_path)

    return dti_path, mesh_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map DTI tensors to FEM mesh nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source: either real files or synthetic
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dti", type=str, help="Path to DTI NIfTI file")
    source.add_argument("--synthetic", action="store_true",
                        help="Generate and use synthetic data")

    parser.add_argument("--mesh", type=str, help="Path to mesh XDMF file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for interpolated tensors")
    parser.add_argument("--epsilon", type=float, default=1e-12,
                        help="Eigenvalue clamping threshold (default: 1e-12)")
    parser.add_argument("--format", dest="output_format", default="h5",
                        choices=["h5", "npy"],
                        help="Output format (default: h5)")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate output (default: enabled)")
    parser.add_argument("--no-validate", action="store_false", dest="validate")
    parser.add_argument("--grid-shape", type=int, nargs=3, default=[32, 32, 32],
                        help="Synthetic grid shape (default: 32 32 32)")
    parser.add_argument("--n-nodes", type=int, default=5000,
                        help="Synthetic mesh node count (default: 5000)")

    args = parser.parse_args()

    if args.synthetic:
        with tempfile.TemporaryDirectory(prefix="dti_mapper_") as tmpdir:
            tmpdir = Path(tmpdir)
            dti_path, mesh_path = create_synthetic_data(
                tmpdir,
                grid_shape=tuple(args.grid_shape),
                n_mesh_nodes=args.n_nodes,
            )
            _run_pipeline(dti_path, mesh_path, args)
    else:
        if args.mesh is None:
            parser.error("--mesh is required when using real data (not --synthetic)")
        _run_pipeline(args.dti, args.mesh, args)


def _run_pipeline(dti_path: str | Path, mesh_path: str | Path, args: argparse.Namespace) -> None:
    """Run the mapping pipeline."""
    from dti_mapper.pipeline import map_tensors_to_mesh

    result = map_tensors_to_mesh(
        dti_path=str(dti_path),
        mesh_path=str(mesh_path),
        output_path=args.output,
        epsilon=args.epsilon,
        validate=args.validate,
        output_format=args.output_format,
    )

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    if result.validation and not result.validation.get("all_valid", False):
        logger.error("Tensor validation FAILED — check output quality")
        sys.exit(1)


if __name__ == "__main__":
    main()
