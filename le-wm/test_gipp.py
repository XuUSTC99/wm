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


def _identity_projection(physics, state_dim=4, **kwargs):
    return GaugeInvariantPhysicsProjection(
        torch.eye(state_dim), torch.zeros(state_dim), torch.eye(state_dim),
        physics=physics, **kwargs)


def test_observed_velocity_expert_uses_position_difference():
    p = _identity_projection("observed_velocity")
    history = torch.tensor([[[0.0, 10.0, 7.0, 7.0],
                             [2.0, 12.0, 9.0, 9.0]]])
    target = p.physics_target(history)
    expected = torch.tensor([[4.0, 14.0, 2.0, 2.0]])
    assert torch.equal(target, expected)


def test_constant_acceleration_expert_extrapolates_velocity_change():
    p = _identity_projection(
        "constant_acceleration", acceleration_clip=0.0)
    history = torch.tensor([[[0.0, 10.0, 1.0, 1.0],
                             [1.0, 11.0, 2.0, 2.0]]])
    target = p.physics_target(history)
    expected = torch.tensor([[3.5, 13.5, 3.0, 3.0]])
    assert torch.equal(target, expected)


def test_damped_velocity_expert_uses_trapezoidal_position_step():
    p = _identity_projection("damped_velocity", damping=0.5)
    history = torch.tensor([[[10.0, 20.0, 2.0, -2.0]]])
    target = p.physics_target(history)
    expected = torch.tensor([[11.5, 18.5, 1.0, -1.0]])
    assert torch.equal(target, expected)


def test_equal_mass_collision_expert_resolves_within_step_impact():
    p = _identity_projection(
        "elastic_collision_equal_mass", collision_distance=2.0)
    history = torch.tensor([[[0.0, 3.0, 1.0, -1.0]]])
    target = p.physics_target(history)
    expected = torch.tensor([[0.0, 3.0, -1.0, 1.0]])
    assert torch.allclose(target, expected)


def test_equal_mass_collision_expert_keeps_separating_objects_free():
    p = _identity_projection(
        "elastic_collision_equal_mass", collision_distance=2.0)
    history = torch.tensor([[[0.0, 3.0, -1.0, 1.0]]])
    target = p.physics_target(history)
    expected = torch.tensor([[-1.0, 4.0, -1.0, 1.0]])
    assert torch.equal(target, expected)


def test_equal_mass_collision_expert_supports_two_2d_objects():
    p = _identity_projection(
        "elastic_collision_equal_mass", state_dim=8,
        collision_distance=2.0)
    history = torch.tensor([[[
        0.0, 5.0, 3.0, 5.0,
        1.0, 0.5, -1.0, 0.5,
    ]]])
    target = p.physics_target(history)
    expected = torch.tensor([[
        0.0, 5.5, 3.0, 5.5,
        -1.0, 0.5, 1.0, 0.5,
    ]])
    assert torch.allclose(target, expected)

def test_temporal_decoder_transports_only_candidate_latent():
    identity = torch.eye(4)
    history_weight = torch.cat([0.1 * identity, -0.2 * identity], dim=1)
    projection = GaugeInvariantPhysicsProjection(
        identity, torch.zeros(4), identity,
        history_weight=history_weight, eps=1e-7,
    )
    history = torch.randn(2, 3, 4)
    predicted = torch.randn(2, 4)
    output = projection(predicted, history)
    assert projection.temporal_history_size == 3
    assert torch.allclose(
        projection.decode_candidate(output, history),
        projection.physics_target(history),
        atol=2e-5,
        rtol=2e-5,
    )
