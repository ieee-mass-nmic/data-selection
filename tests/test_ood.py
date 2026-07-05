"""OOD detection + calibration head smoke tests."""

from __future__ import annotations

import numpy as np
import torch

from pcu_select.ood.calibration import (
    CalibrationHead,
    apply_calibration,
    fit_calibration,
    fit_ood_stats,
    is_ood,
    load_calibration,
    save_calibration,
)


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


def test_calibration_save_load_roundtrip_and_apply(tmp_path):
    rng = np.random.default_rng(0)
    n = 6
    head = CalibrationHead(in_dim=4)
    # give it non-trivial weights so the roundtrip is meaningful
    with torch.no_grad():
        head.linear.weight.copy_(torch.randn_like(head.linear.weight))
        head.linear.bias.copy_(torch.randn_like(head.linear.bias))

    mu = rng.normal(size=n).astype(np.float32)
    z_x = rng.normal(size=(n, 2)).astype(np.float32)
    z_p = rng.normal(size=(1, 1)).astype(np.float32)  # broadcast over batch
    z_t = rng.normal(size=(1, 1)).astype(np.float32)

    expected = apply_calibration(head, mu=mu, z_x=z_x, z_p=z_p, z_t=z_t, device="cpu")

    path = save_calibration(head, tmp_path / "calib.pt")
    assert path.exists()
    reloaded = load_calibration(path, device="cpu")
    got = apply_calibration(reloaded, mu=mu, z_x=z_x, z_p=z_p, z_t=z_t, device="cpu")

    assert got.shape == (n,)
    assert np.allclose(expected, got, atol=1e-6)
