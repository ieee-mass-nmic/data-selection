"""OOD detection + calibration head smoke tests."""

from __future__ import annotations

import numpy as np
import torch

from pcu_select.ood.calibration import CalibrationHead, fit_calibration, fit_ood_stats, is_ood


def test_ood_threshold_quantile():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 16)).astype(np.float32)
    stats = fit_ood_stats(z, quantile=0.95)
    # ~5% of training points should be > threshold
    diff = z - stats.mu
    d2 = np.einsum("nd,de,ne->n", diff, stats.sigma_inv, diff)
    frac_over = float((d2 > stats.threshold).mean())
    assert 0.02 <= frac_over <= 0.10


def test_is_ood_flags_far_point():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 8)).astype(np.float32)
    stats = fit_ood_stats(z, quantile=0.95)
    near = z.mean(axis=0)
    far = near + 50.0
    assert not is_ood(near, stats)
    assert is_ood(far, stats)


def test_calibration_head_runs_one_step():
    head = CalibrationHead(in_dim=4)
    mu_hat = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    u_hi = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    z_x = np.zeros((3, 2), dtype=np.float32)
    z_p = np.zeros((1, 1), dtype=np.float32)
    z_t = np.zeros((3, 1), dtype=np.float32)
    fit_calibration(head=head, mu_hat=mu_hat, u_hi=u_hi,
                    z_x=z_x, z_p=z_p, z_t=z_t,
                    epochs=5, device="cpu")
    # head shouldn't have exploded
    for p in head.parameters():
        assert torch.isfinite(p).all()
