import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gipp import GaugeInvariantPhysicsProjection


def test_projection_hits_state_and_is_basis_equivariant():
    torch.manual_seed(7)
    d, s = 8, 4
    w = torch.randn(s, d)
    sigma = torch.randn(d, d); sigma = sigma @ sigma.T + 0.2 * torch.eye(d)
    b = torch.randn(s)
    zhist = torch.randn(2, 3, d)
    zpred = torch.randn(2, d)
    p = GaugeInvariantPhysicsProjection(w, b, sigma, eps=1e-7)
    out = p(zpred, zhist)
    assert torch.allclose(p.decode(out), p.physics_target(zhist), atol=2e-4, rtol=2e-4)

    a = torch.randn(d, d) + 2.0 * torch.eye(d)
    ai = torch.linalg.inv(a)
    w2 = w @ ai
    sigma2 = a @ sigma @ a.T
    p2 = GaugeInvariantPhysicsProjection(w2, b, sigma2, eps=1e-7)
    out2 = p2(zpred @ a.T, zhist @ a.T)
    assert torch.allclose(out2, out @ a.T, atol=2e-3, rtol=2e-3)


def test_alpha_zero_is_black_box_control():
    w = torch.eye(4)
    p = GaugeInvariantPhysicsProjection(w, torch.zeros(4), torch.eye(4), alpha=0.0)
    pred, hist = torch.randn(3, 4), torch.randn(3, 2, 4)
    assert torch.equal(p(pred, hist), pred)


def test_innovation_gate_suppresses_small_residuals():
    p = GaugeInvariantPhysicsProjection(
        torch.eye(4), torch.zeros(4), torch.eye(4), alpha=1.0,
        state_scale=torch.ones(4), gate="innovation",
        gate_threshold=1.0, gate_temperature=0.1,
    )
    small = p.correction_strength(torch.zeros(2, 4))
    large = p.correction_strength(torch.full((2, 4), 3.0))
    assert torch.all(small < 1e-3)
    assert torch.all(large > 0.999)
