"""Tests for trilinear interpolation of DTI tensors onto FEM meshes."""

from __future__ import annotations

import numpy as np
import pytest


class TestTrilinearWeights:
    """Test trilinear weight computation."""

    def test_weights_at_grid_center(self) -> None:
        """All 8 weights should be 1/8 at voxel center."""
        from dti_mapper import compute_trilinear_weights

        inv_affine = np.eye(4, dtype=np.float64)
        points = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)

        indices, weights = compute_trilinear_weights(points, inv_affine, (4, 4, 4))
        np.testing.assert_allclose(weights[0], 0.125, atol=1e-12)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)

    def test_weights_at_vertex(self) -> None:
        """One weight should be 1.0 at an exact grid vertex."""
        from dti_mapper import compute_trilinear_weights

        inv_affine = np.eye(4, dtype=np.float64)
        points = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)

        indices, weights = compute_trilinear_weights(points, inv_affine, (4, 4, 4))
        assert np.max(weights) == pytest.approx(1.0, abs=1e-12)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-12)

    def test_weights_sum_to_one(self, synthetic_mesh_nodes: np.ndarray) -> None:
        """Weights for every node should sum to 1."""
        from dti_mapper import compute_trilinear_weights

        inv_affine = np.eye(4, dtype=np.float64)
        indices, weights = compute_trilinear_weights(
            synthetic_mesh_nodes, inv_affine, (16, 16, 16)
        )

        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_scaled_affine(self) -> None:
        """Non-identity affine (2mm voxel spacing) should produce correct indices."""
        from dti_mapper import compute_trilinear_weights

        # 2mm voxel spacing: physical = 2 * voxel → inv_affine scales by 0.5
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        inv_affine = np.linalg.inv(affine)

        # Physical point (3.0, 3.0, 3.0) → voxel (1.5, 1.5, 1.5) → center of cube
        points = np.array([[3.0, 3.0, 3.0]], dtype=np.float64)
        indices, weights = compute_trilinear_weights(points, inv_affine, (4, 4, 4))

        np.testing.assert_allclose(weights[0], 0.125, atol=1e-12)


class TestInterpolation:
    """Test full interpolation pipeline."""

    def test_uniform_field_returns_same_tensor(
        self,
        uniform_tensor_volume: tuple[np.ndarray, np.ndarray],
        synthetic_mesh_nodes: np.ndarray,
    ) -> None:
        """Interpolating a uniform field should return the same tensor everywhere."""
        from dti_mapper import (
            compute_trilinear_weights,
            interpolate_fem_points,
            precompute_logs,
        )

        tensors, affine = uniform_tensor_volume
        expected_tensor = tensors[0, 0, 0]

        # Flatten and precompute logs
        flat = tensors.reshape(-1, 3, 3)
        log_tensors = precompute_logs(flat)

        # Compute weights
        inv_affine = np.linalg.inv(affine)
        indices, weights = compute_trilinear_weights(
            synthetic_mesh_nodes, inv_affine, tensors.shape[:3]
        )

        # Interpolate
        result_flat = interpolate_fem_points(log_tensors, indices, weights)
        result = result_flat.reshape(-1, 3, 3)

        for i in range(len(result)):
            np.testing.assert_allclose(
                result[i], expected_tensor, atol=1e-8,
                err_msg=f"Tensor at node {i} differs from expected",
            )

    def test_gradient_field_monotonic(
        self,
        gradient_tensor_volume: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """For a tensor field with gradient along x, interpolated tensors
        at increasing x should have increasing (0,0) component."""
        from dti_mapper import (
            compute_trilinear_weights,
            interpolate_fem_points,
            precompute_logs,
        )

        tensors, affine = gradient_tensor_volume

        # Sample 10 points along x-axis at y=z=8
        x_coords = np.linspace(1.0, 14.0, 10)
        points = np.column_stack([
            x_coords,
            np.full(10, 8.0),
            np.full(10, 8.0),
        ]).astype(np.float64)

        flat = tensors.reshape(-1, 3, 3)
        log_tensors = precompute_logs(flat)
        inv_affine = np.linalg.inv(affine)
        indices, weights = compute_trilinear_weights(
            points, inv_affine, tensors.shape[:3]
        )
        result = interpolate_fem_points(log_tensors, indices, weights).reshape(-1, 3, 3)

        # D_xx should be monotonically increasing
        d_xx = result[:, 0, 0]
        for i in range(1, len(d_xx)):
            assert d_xx[i] >= d_xx[i - 1] - 1e-10, (
                f"D_xx not monotonic at position {i}: {d_xx[i]} < {d_xx[i-1]}"
            )

    def test_interpolated_tensors_are_spd(
        self,
        uniform_tensor_volume: tuple[np.ndarray, np.ndarray],
        synthetic_mesh_nodes: np.ndarray,
    ) -> None:
        """All interpolated tensors must be SPD."""
        from dti_mapper import (
            compute_trilinear_weights,
            interpolate_fem_points,
            precompute_logs,
            validate_tensors,
        )

        tensors, affine = uniform_tensor_volume
        flat = tensors.reshape(-1, 3, 3)
        log_tensors = precompute_logs(flat)
        inv_affine = np.linalg.inv(affine)
        indices, weights = compute_trilinear_weights(
            synthetic_mesh_nodes, inv_affine, tensors.shape[:3]
        )
        result = interpolate_fem_points(log_tensors, indices, weights).reshape(-1, 3, 3)

        report = validate_tensors(result)
        assert report["all_valid"], f"Validation failed: {report['summary']}"

    def test_interpolation_at_grid_vertex(self) -> None:
        """Interpolation at an exact grid vertex should return that voxel's tensor."""
        from dti_mapper import (
            compute_trilinear_weights,
            interpolate_fem_points,
            precompute_logs,
        )

        # 4×4×4 grid with distinct diagonal tensors
        tensors = np.zeros((4, 4, 4, 3, 3), dtype=np.float64)
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    tensors[ix, iy, iz] = np.diag([1.0 + ix, 1.0 + iy, 1.0 + iz])

        flat = tensors.reshape(-1, 3, 3)
        log_tensors = precompute_logs(flat)
        inv_affine = np.eye(4, dtype=np.float64)

        # Query at vertex (2, 1, 3)
        points = np.array([[2.0, 1.0, 3.0]], dtype=np.float64)
        indices, weights = compute_trilinear_weights(points, inv_affine, (4, 4, 4))
        result = interpolate_fem_points(log_tensors, indices, weights).reshape(-1, 3, 3)

        expected = np.diag([3.0, 2.0, 4.0])
        np.testing.assert_allclose(result[0], expected, atol=1e-8)
