"""dti_mapper — Log-Euclidean DTI tensor interpolation onto FEM meshes.

This package provides a high-performance pipeline for mapping diffusion tensor
imaging (DTI) data from regular voxel grids to unstructured finite element
mesh nodes, using the log-Euclidean framework to guarantee interpolated tensors
remain symmetric positive definite (SPD).

The computational core is implemented in Rust (via PyO3) for parallel performance.
Python modules handle I/O, validation, and orchestration.

Quick start
-----------
>>> from dti_mapper import map_tensors_to_mesh
>>> result = map_tensors_to_mesh(
...     dti_path="data/dti.nii.gz",
...     mesh_path="mesh/mesh.xdmf",
...     output_path="tensors/tensors.h5",
... )
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export Rust functions
from dti_mapper._rust import (
    DEFAULT_EPSILON,
    DEFAULT_SYMMETRY_TOL,
    compute_trilinear_weights_py as compute_trilinear_weights,
    interpolate_fem_points_py as interpolate_fem_points,
    precompute_logs,
    tensor_exp_single,
    tensor_log_single,
    validate_tensors_py as validate_tensors,
)

# Re-export Python I/O
from dti_mapper.io import (
    load_dti_nifti,
    load_mesh_nodes,
    save_tensors_h5,
    save_tensors_npy,
)
from dti_mapper.pipeline import map_tensors_to_mesh
from dti_mapper.validate import validate_tensor_output

__all__ = [
    # Core Rust functions
    "precompute_logs",
    "interpolate_fem_points",
    "compute_trilinear_weights",
    "validate_tensors",
    "tensor_log_single",
    "tensor_exp_single",
    "DEFAULT_EPSILON",
    "DEFAULT_SYMMETRY_TOL",
    # Python I/O
    "load_dti_nifti",
    "load_mesh_nodes",
    "save_tensors_h5",
    "save_tensors_npy",
    # High-level pipeline
    "map_tensors_to_mesh",
    "validate_tensor_output",
]
