"""Gauge-equivariant physics residual transport for latent rollouts."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GaugeInvariantPhysicsProjection(nn.Module):
    """Covariance-metric transport of a decoded state residual into a latent."""

    def __init__(self, weight, bias, covariance, alpha=1.0, eps=1e-4,
                 physics="constant_velocity", gravity=None, state_scale=None,
                 gate="constant", gate_threshold=0.15, gate_temperature=0.05,
                 horizon_start=12.0, horizon_temperature=2.0):
        super().__init__()
        weight = torch.as_tensor(weight, dtype=torch.float32)
        bias = torch.as_tensor(bias, dtype=torch.float32)
        covariance = torch.as_tensor(covariance, dtype=torch.float32)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("GIPP decoder weight/bias shapes are inconsistent")
        if covariance.shape != (weight.shape[1], weight.shape[1]):
            raise ValueError("GIPP covariance does not match latent dimension")
        if weight.shape[0] % 2:
            raise ValueError("GIPP state must contain equal position/velocity blocks")
        if gate not in {"constant", "innovation", "horizon", "innovation_horizon"}:
            raise ValueError(f"unknown GIPP gate: {gate}")

        state_cov = weight @ covariance @ weight.T
        eye = torch.eye(weight.shape[0], dtype=weight.dtype)
        gain = torch.linalg.solve(state_cov + float(eps) * eye,
                                  weight @ covariance).T
        scale = torch.ones(weight.shape[0]) if state_scale is None else torch.as_tensor(
            state_scale, dtype=torch.float32)
        if scale.shape != bias.shape:
            raise ValueError("state_scale must have one value per decoded state dimension")

        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("gain", gain)
        self.register_buffer("state_scale", scale.clamp_min(1e-6))
        self.register_buffer(
            "gravity",
            torch.as_tensor([] if gravity is None else gravity, dtype=torch.float32),
        )
        self.alpha = float(alpha)
        self.physics = str(physics)
        self.gate = str(gate)
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.horizon_start = float(horizon_start)
        self.horizon_temperature = float(horizon_temperature)

    @classmethod
    def from_npz(cls, path, **kwargs):
        data = np.load(Path(path))
        gravity = kwargs.pop("gravity", None)
        if gravity is None and "gravity" in data:
            gravity = data["gravity"]
        state_scale = kwargs.pop("state_scale", None)
        if state_scale is None and "state_scale" in data:
            state_scale = data["state_scale"]
        return cls(data["weight"], data["bias"], data["covariance"],
                   gravity=gravity, state_scale=state_scale, **kwargs)

    @property
    def pos_dim(self):
        return self.weight.shape[0] // 2

    def decode(self, latent):
        return latent @ self.weight.T + self.bias

    def physics_target(self, history):
        if history.size(-2) < 1:
            raise ValueError("GIPP needs at least one history latent")
        state = self.decode(history[..., -1, :])
        p, v = state.split(self.pos_dim, dim=-1)
        if self.physics == "constant_velocity":
            accel = torch.zeros_like(v)
        elif self.physics == "gravity":
            if self.gravity.numel() != self.pos_dim:
                raise ValueError("gravity physics requires one value per position dim")
            accel = self.gravity.to(v).expand_as(v)
        else:
            raise ValueError(f"unknown GIPP physics mode: {self.physics}")
        return torch.cat([p + v + 0.5 * accel, v + accel], dim=-1)

    def correction_strength(self, residual, alpha=None, rollout_step=None):
        maximum = self.alpha if alpha is None else float(alpha)
        score = ((residual / self.state_scale.to(residual)) ** 2).mean(
            dim=-1, keepdim=True).sqrt()
        self.last_gate_score = score.detach()
        strength = torch.as_tensor(maximum, dtype=residual.dtype, device=residual.device)
        if self.gate in {"innovation", "innovation_horizon"}:
            temperature = max(self.gate_temperature, 1e-6)
            strength = strength * torch.sigmoid(
                (score - self.gate_threshold) / temperature)
        if self.gate in {"horizon", "innovation_horizon"}:
            if rollout_step is None:
                raise ValueError("horizon-gated GIPP requires rollout_step")
            step = torch.as_tensor(rollout_step, dtype=residual.dtype,
                                   device=residual.device)
            temperature = max(self.horizon_temperature, 1e-6)
            strength = strength * torch.sigmoid(
                (step - self.horizon_start) / temperature)
        return strength

    def forward(self, predicted, history, alpha=None, rollout_step=None):
        target = self.physics_target(history)
        residual = target - self.decode(predicted)
        strength = self.correction_strength(
            residual, alpha=alpha, rollout_step=rollout_step)
        correction = residual @ self.gain.T
        return predicted + strength * correction
