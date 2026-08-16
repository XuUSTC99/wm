"""Gauge-equivariant physics residual transport for latent rollouts."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GaugeInvariantPhysicsProjection(nn.Module):
    """Covariance-metric transport of a decoded state residual into a latent."""

    def __init__(self, weight, bias, covariance, alpha=1.0, eps=1e-4,
                 physics="constant_velocity", gravity=None, state_scale=None,
                 history_weight=None, damping=0.95, acceleration_clip=3.0,
                 collision_distance=2.5, gate="constant",
                 gate_threshold=0.15, gate_temperature=0.05,
                 horizon_start=12.0, horizon_temperature=2.0):
        super().__init__()
        weight = torch.as_tensor(weight, dtype=torch.float32)
        bias = torch.as_tensor(bias, dtype=torch.float32)
        covariance = torch.as_tensor(covariance, dtype=torch.float32)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("GIPP decoder weight/bias shapes are inconsistent")
        if covariance.shape != (weight.shape[1], weight.shape[1]):
            raise ValueError("GIPP covariance does not match latent dimension")
        history_weight = torch.as_tensor(
            np.empty((weight.shape[0], 0)) if history_weight is None
            else history_weight,
            dtype=torch.float32,
        )
        if (history_weight.ndim != 2
                or history_weight.shape[0] != weight.shape[0]
                or history_weight.shape[1] % weight.shape[1]):
            raise ValueError(
                "history_weight must contain whole latent blocks per state")
        if weight.shape[0] % 2:
            raise ValueError("GIPP state must contain equal position/velocity blocks")
        if gate not in {"constant", "innovation", "horizon", "innovation_horizon"}:
            raise ValueError(f"unknown GIPP gate: {gate}")
        if not 0.0 <= float(damping) <= 1.5:
            raise ValueError("damping must be in [0, 1.5]")
        if float(collision_distance) <= 0:
            raise ValueError("collision_distance must be positive")

        state_cov = weight @ covariance @ weight.T
        eye = torch.eye(weight.shape[0], dtype=weight.dtype)
        gain = torch.linalg.solve(state_cov + float(eps) * eye,
                                  weight @ covariance).T
        scale = torch.ones(weight.shape[0]) if state_scale is None else torch.as_tensor(
            state_scale, dtype=torch.float32)
        if scale.shape != bias.shape:
            raise ValueError("state_scale must have one value per decoded state dimension")

        self.register_buffer("weight", weight)
        self.register_buffer("history_weight", history_weight)
        self.register_buffer("bias", bias)
        self.register_buffer("gain", gain)
        self.register_buffer("state_scale", scale.clamp_min(1e-6))
        self.register_buffer(
            "gravity",
            torch.as_tensor([] if gravity is None else gravity, dtype=torch.float32),
        )
        self.alpha = float(alpha)
        self.physics = str(physics)
        self.damping = float(damping)
        self.acceleration_clip = float(acceleration_clip)
        self.collision_distance = float(collision_distance)
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
        history_weight = kwargs.pop("history_weight", None)
        if history_weight is None and "history_weight" in data:
            history_weight = data["history_weight"]
        return cls(
            data["weight"], data["bias"], data["covariance"],
            gravity=gravity, state_scale=state_scale,
            history_weight=history_weight, **kwargs,
        )

    @property
    def pos_dim(self):
        return self.weight.shape[0] // 2

    @property
    def temporal_history_size(self):
        history_weight = getattr(self, "history_weight", None)
        if history_weight is None or history_weight.numel() == 0:
            return 1
        return 1 + history_weight.shape[1] // self.weight.shape[1]

    def decode(self, latent):
        return latent @ self.weight.T + self.bias

    def decode_history(self, history):
        required = self.temporal_history_size
        if history.size(-2) < required:
            raise ValueError(
                f"temporal GIPP needs at least {required} history latents")
        current = self.decode(history[..., -1, :])
        if required == 1:
            return current
        prefix = history[..., -required:-1, :].flatten(-2)
        return current + prefix @ self.history_weight.T

    def decode_candidate(self, predicted, history):
        required = self.temporal_history_size
        if required == 1:
            return self.decode(predicted)
        if history.size(-2) < required - 1:
            raise ValueError(
                f"temporal GIPP needs {required - 1} prefix latents")
        prefix = history[..., -(required - 1):, :].flatten(-2)
        return self.decode(predicted) + prefix @ self.history_weight.T

    def physics_target(self, history):
        if history.size(-2) < self.temporal_history_size:
            raise ValueError(
                f"GIPP needs {self.temporal_history_size} history latents")
        state = self.decode_history(history)
        p, v = state.split(self.pos_dim, dim=-1)
        if self.physics == "constant_velocity":
            return torch.cat([p + v, v], dim=-1)
        elif self.physics == "gravity":
            if self.gravity.numel() != self.pos_dim:
                raise ValueError("gravity physics requires one value per position dim")
            accel = self.gravity.to(v).expand_as(v)
            return torch.cat([p + v + 0.5 * accel, v + accel], dim=-1)
        elif self.physics == "observed_velocity":
            if history.size(-2) < self.temporal_history_size + 1:
                raise ValueError(
                    "observed_velocity physics needs two decoded states")
            previous = self.decode_history(history[..., :-1, :])
            p_prev = previous[..., :self.pos_dim]
            observed_v = p - p_prev
            return torch.cat([p + observed_v, observed_v], dim=-1)
        elif self.physics == "constant_acceleration":
            if history.size(-2) < self.temporal_history_size + 1:
                raise ValueError(
                    "constant_acceleration physics needs two decoded states")
            previous = self.decode_history(history[..., :-1, :])
            v_prev = previous[..., self.pos_dim:]
            accel = v - v_prev
            if self.acceleration_clip > 0:
                limit = (
                    self.acceleration_clip
                    * self.state_scale[self.pos_dim:].to(accel)
                )
                accel = torch.maximum(torch.minimum(accel, limit), -limit)
            return torch.cat([p + v + 0.5 * accel, v + accel], dim=-1)
        elif self.physics == "damped_velocity":
            v_next = self.damping * v
            return torch.cat([p + 0.5 * (v + v_next), v_next], dim=-1)
        elif self.physics == "elastic_collision_equal_mass":
            if self.pos_dim % 2:
                raise ValueError(
                    "elastic_collision_equal_mass requires two equal-size "
                    "object position blocks")
            coordinate_dim = self.pos_dim // 2
            object_p = p.reshape(*p.shape[:-1], 2, coordinate_dim)
            object_v = v.reshape(*v.shape[:-1], 2, coordinate_dim)
            separation = object_p[..., 1, :] - object_p[..., 0, :]
            distance = torch.linalg.vector_norm(separation, dim=-1)
            normal = separation / distance.clamp_min(1e-6)[..., None]
            relative_v = object_v[..., 0, :] - object_v[..., 1, :]
            closing_speed = (relative_v * normal).sum(dim=-1)
            gap = distance - self.collision_distance
            safe_speed = closing_speed.clamp_min(1e-6)
            impact_time = (gap / safe_speed).clamp(0.0, 1.0)
            collides = (closing_speed > 1e-6) & (gap <= closing_speed)
            normal_impulse = closing_speed[..., None] * normal
            collision_v = torch.stack(
                [object_v[..., 0, :] - normal_impulse,
                 object_v[..., 1, :] + normal_impulse],
                dim=-2,
            )
            v_next = torch.where(
                collides[..., None, None], collision_v, object_v)
            collision_p = object_p + object_v * impact_time[..., None, None]
            collision_next_p = collision_p + collision_v * (
                1.0 - impact_time[..., None, None])
            free_next_p = object_p + object_v
            p_next = torch.where(
                collides[..., None, None], collision_next_p, free_next_p)
            return torch.cat(
                [p_next.flatten(-2), v_next.flatten(-2)], dim=-1)
        else:
            raise ValueError(f"unknown GIPP physics mode: {self.physics}")

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
        residual = target - self.decode_candidate(predicted, history)
        strength = self.correction_strength(
            residual, alpha=alpha, rollout_step=rollout_step)
        correction = residual @ self.gain.T
        return predicted + strength * correction
