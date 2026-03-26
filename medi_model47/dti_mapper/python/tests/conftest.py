"""Shared test fixtures for dti_mapper tests."""

from __future__ import annotations

import numpy as np
import pytest


def _make_spd(eigenvalues: tuple[float, ...] = (2.0, 1.0, 0.5),
              seed: int = 42) -> np.ndarray:
    """Create a random SPD tensor with given eigenvalues."""
    rng = np.random.default_rng(seed)
    # Random orthogonal matrix via QR decomposition
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    D = np.diag(eigenvalues)
    return Q @ D @ Q.T


@pytest.fixture
def spd_tensor() -> np.ndarray:
    """Single SPD tensor (3, 3)."""
    return _make_spd()


@pytest.fixture
def identity_tensor() -> np.ndarray:
    """Identity tensor (3, 3)."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def uniform_tensor_volume() -> tuple[np.ndarray, np.ndarray]:
    """Uniform tensor field (16, 16, 16, 3, 3) with identity affine.

    Returns (tensors, affine).
    """
    t = _make_spd(eigenvalues=(1.5, 1.0, 0.5), seed=7)
    shape = (16, 16, 16)
    tensors = np.tile(t, (*shape, 1, 1))
    affine = np.eye(4, dtype=np.float64)
    return tensors, affine


@pytest.fixture
def gradient_tensor_volume() -> tuple[np.ndarray, np.ndarray]:
    """Tensor field with a gradient along x-axis.

    Eigenvalues increase linearly from (1, 1, 1) to (3, 1, 1) along x.
    Returns (tensors, affine).
    """
    shape = (16, 16, 16)
    tensors = np.zeros((*shape, 3, 3), dtype=np.float64)

    for ix in range(shape[0]):
        scale = 1.0 + 2.0 * ix / (shape[0] - 1)
        tensors[ix, :, :] = np.diag([scale, 1.0, 1.0])

    affine = np.eye(4, dtype=np.float64)
    return tensors, affine


@pytest.fixture
def synthetic_mesh_nodes() -> np.ndarray:
    """100 random points within [1, 14]^3 (inside a 16^3 grid)."""
    rng = np.random.default_rng(123)
    return rng.uniform(1.0, 14.0, size=(100, 3))
