"""Tests for the Rust tensor_log / tensor_exp bindings."""

from __future__ import annotations

import numpy as np
import pytest


class TestTensorLogExp:
    """Test matrix logarithm and exponential roundtrip."""

    def test_log_exp_roundtrip(self, spd_tensor: np.ndarray) -> None:
        from dti_mapper import tensor_exp_single, tensor_log_single

        log_t = tensor_log_single(spd_tensor)
        recovered = tensor_exp_single(log_t)
        np.testing.assert_allclose(recovered, spd_tensor, atol=1e-10)

    def test_log_identity_is_zero(self, identity_tensor: np.ndarray) -> None:
        from dti_mapper import tensor_log_single

        log_id = tensor_log_single(identity_tensor)
        np.testing.assert_allclose(log_id, np.zeros((3, 3)), atol=1e-12)

    def test_exp_zero_is_identity(self) -> None:
        from dti_mapper import tensor_exp_single

        zero = np.zeros((3, 3), dtype=np.float64)
        exp_zero = tensor_exp_single(zero)
        np.testing.assert_allclose(exp_zero, np.eye(3), atol=1e-12)

    def test_log_preserves_symmetry(self, spd_tensor: np.ndarray) -> None:
        from dti_mapper import tensor_log_single

        log_t = tensor_log_single(spd_tensor)
        np.testing.assert_allclose(log_t, log_t.T, atol=1e-14)

    def test_exp_result_is_spd(self) -> None:
        from dti_mapper import tensor_exp_single

        sym = np.array([[1.0, 0.5, 0.2],
                        [0.5, 2.0, 0.3],
                        [0.2, 0.3, 1.5]], dtype=np.float64)
        exp_t = tensor_exp_single(sym)

        eigenvalues = np.linalg.eigvalsh(exp_t)
        assert np.all(eigenvalues > 0), f"Not SPD: eigenvalues = {eigenvalues}"

    def test_batch_precompute_consistent_with_single(self) -> None:
        """precompute_logs on a batch should match individual tensor_log calls."""
        from dti_mapper import precompute_logs, tensor_log_single

        rng = np.random.default_rng(42)
        n = 50
        tensors = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            A = rng.standard_normal((3, 3))
            tensors[i] = A @ A.T + np.eye(3) * 0.01

        batch_logs = precompute_logs(tensors)  # (N, 9)

        for i in range(n):
            single_log = tensor_log_single(tensors[i])
            expected_flat = single_log.reshape(-1)
            np.testing.assert_allclose(batch_logs[i], expected_flat, atol=1e-12)


class TestTensorValidation:
    """Test the Rust validation function."""

    def test_valid_spd_tensors(self) -> None:
        from dti_mapper import validate_tensors

        tensors = np.zeros((10, 3, 3), dtype=np.float64)
        for i in range(10):
            tensors[i] = np.diag([2.0 + i * 0.1, 1.0, 0.5])

        report = validate_tensors(tensors)
        assert report["all_valid"] is True
        assert report["n_spd"] == 10
        assert report["min_eigenvalue"] > 0

    def test_non_spd_detected(self) -> None:
        from dti_mapper import validate_tensors

        # Tensor with negative eigenvalue
        tensors = np.array([[[2.0, 0, 0], [0, -0.5, 0], [0, 0, 1.0]]], dtype=np.float64)
        report = validate_tensors(tensors)
        assert report["all_valid"] is False
        assert report["n_spd"] == 0

    def test_nan_detected(self) -> None:
        from dti_mapper import validate_tensors

        tensors = np.array([[[1.0, 0, 0], [0, float("nan"), 0], [0, 0, 1.0]]], dtype=np.float64)
        report = validate_tensors(tensors)
        assert report["n_finite"] == 0
