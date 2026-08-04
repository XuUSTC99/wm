"""Attach a frozen GIPP module to a trained JEPA object without retraining."""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "le-wm"))
from gipp import GaugeInvariantPhysicsProjection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--physics", choices=["constant_velocity", "gravity"], required=True)
    ap.add_argument("--gate", choices=["constant", "innovation", "horizon", "innovation_horizon"], default="constant")
    ap.add_argument("--gate-threshold", type=float, default=0.15)
    ap.add_argument("--gate-temperature", type=float, default=0.05)
    ap.add_argument("--horizon-start", type=float, default=12.0)
    ap.add_argument("--horizon-temperature", type=float, default=2.0)
    ap.add_argument("--shadow", action="store_true")
    args = ap.parse_args()

    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("expected a serialized JEPA model object")
    model.gipp = GaugeInvariantPhysicsProjection.from_npz(
        args.state, alpha=args.alpha, physics=args.physics, gate=args.gate,
        gate_threshold=args.gate_threshold, gate_temperature=args.gate_temperature,
        horizon_start=args.horizon_start, horizon_temperature=args.horizon_temperature,
    )
    model.gipp.shadow = bool(args.shadow)
    if hasattr(model, "use_action"):
        model.use_action = False
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"已保存零训练 GIPP 模型：{out}")


if __name__ == "__main__":
    main()
