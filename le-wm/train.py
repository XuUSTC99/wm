import os
import re
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from gipp import GaugeInvariantPhysicsProjection
from module import ARPredictor, Embedder, MLP, SIGReg, SecondOrderDynamics
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def _remap_old_vit_keys(sd):
    """Remap old HuggingFace ViT param names -> the new transformers ViT naming
    used by the current model. Older LeWM checkpoints (e.g. lewm_paper_pusht/
    weights.pt) were saved with `encoder.encoder.layer.N.attention.attention.*`
    style names; newer transformers refactored these to
    `encoder.layers.N.attention.{q,k,v,o}_proj` etc. Without this remap,
    load_state_dict(strict=False) silently drops all 192 transformer-body
    weights and the encoder is left at random init. Pass-through for keys that
    already use the new naming (idempotent)."""
    sub = [
        (r"\.attention\.attention\.query\.", ".attention.q_proj."),
        (r"\.attention\.attention\.key\.", ".attention.k_proj."),
        (r"\.attention\.attention\.value\.", ".attention.v_proj."),
        (r"\.attention\.output\.dense\.", ".attention.o_proj."),
        (r"\.intermediate\.dense\.", ".mlp.fc1."),
        (r"\.output\.dense\.", ".mlp.fc2."),
    ]
    out = {}
    for k, v in sd.items():
        nk = re.sub(r"^encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1.", k)
        for pat, rep in sub:
            nk = re.sub(pat, rep, nk)
        out[nk] = v
    return out


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    aug_cfg = cfg.get("aug", None)

    # Temporal (velocity) augmentation (NEW 2026-07-06): randomly stride the loaded
    # frame sequence to simulate different speeds -> attacks the v-OOD softspot (unseen
    # velocities) that photometric/scale aug can't touch. Requires the data to load
    # EXTRA frames: set data.dataset.num_steps = (history+num_preds) * aug.temporal.
    # Pure-video safe: strides ALL temporal keys (pixels/action/proprio/state) together.
    # Applied first, before appearance/scale, so those act on the strided frames.
    _temporal = int(aug_cfg.get("temporal", 1)) if aug_cfg is not None else 1
    if stage == "fit" and _temporal > 1:
        base_T = ctx_len + n_preds
        T_loaded = batch["pixels"].size(1)
        max_s = min(_temporal, T_loaded // base_T)
        if max_s >= 2:
            s = int(torch.randint(1, max_s + 1, (1,)).item())           # per-batch stride
            span = (base_T - 1) * s + 1
            off = int(torch.randint(0, T_loaded - span + 1, (1,)).item())
            idx = off + torch.arange(base_T, device=batch["pixels"].device) * s
            for k in list(batch.keys()):
                if torch.is_tensor(batch[k]) and batch[k].dim() >= 2 and batch[k].size(1) == T_loaded:
                    batch[k] = batch[k].index_select(1, idx)

    # Appearance-invariance augmentation (NEW 2026-07-05): per-trial (not per-frame,
    # to keep temporal consistency) brightness/contrast jitter on normalized pixels
    # during training. Forces the encoder to be invariant to appearance -> attacks
    # the r/m-OOD softspot (radius/mass change -> appearance change -> biased position
    # encoding) AND shrinks the synthetic->real domain gap (phyworld-trained reps
    # currently transfer to Physion *worse than random*; appearance is the gap).
    # Off by default (aug.appearance=0).
    aug_cfg = cfg.get("aug", None)
    if stage == "fit" and aug_cfg is not None and float(aug_cfg.get("appearance", 0.0)) > 0:
        s = float(aug_cfg["appearance"])
        px = batch["pixels"].float()  # (B, T, C, H, W), imagenet-normalized
        bsz = px.size(0)
        bright = torch.randn(bsz, 1, 1, 1, 1, device=px.device) * s              # per-trial shift
        contrast = (1.0 + torch.randn(bsz, 1, 1, 1, 1, device=px.device) * s).clamp(0.5, 1.5)
        batch["pixels"] = px * contrast + bright

    # Scale-invariance augmentation (NEW 2026-07-06): per-trial random center-zoom on
    # pixels, temporally consistent. Attacks the r/m-OOD softspot DIRECTLY -- radius
    # change is a ball SIZE change, which photometric jitter can't simulate. SAFE only
    # for pure-video training (no proprio target): a geometric transform desyncs proprio
    # labels, so we skip it when structured/consistency losses are active. Off by default.
    _scale_s = float(aug_cfg.get("scale", 0.0)) if aug_cfg is not None else 0.0
    _struct_on = (float((cfg.loss.get("structured", {}) or {}).get("weight", 0.0)) > 0
                  or float((cfg.loss.get("consistency", {}) or {}).get("weight", 0.0)) > 0)
    if stage == "fit" and _scale_s > 0 and not _struct_on:
        px = batch["pixels"].float()
        B, T, C, H, W = px.shape
        sc = 1.0 + (torch.rand(B, device=px.device) * 2 - 1) * _scale_s   # per-trial scale in [1-s,1+s]
        theta = torch.zeros(B, 2, 3, device=px.device, dtype=px.dtype)
        theta[:, 0, 0] = 1.0 / sc                                          # grid uses inverse scale
        theta[:, 1, 1] = 1.0 / sc
        theta = theta.repeat_interleave(T, dim=0)                          # (B*T,2,3)
        x = px.reshape(B * T, C, H, W)
        grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="border")
        batch["pixels"] = x.reshape(B, T, C, H, W)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]
    ctx_action = batch["action"][:, :ctx_len]  # raw action for structured dynamics (ignored if off)

    if cfg.wm.get("free_rollout", False):
        # Free-rollout (no teacher forcing): unroll the predictor autoregressively over
        # the num_preds horizon, feeding each prediction back as the next input. Trains the
        # recursive dynamics to not compound errors (PIWM §4.1) -> directly targets the
        # long-horizon drift that 1-step teacher forcing masks. Needs num_preds>1.
        emb_hist = ctx_emb.clone()
        preds = []
        _gipp = getattr(self.model, "gipp", None)
        _shadow = _gipp is not None and bool(getattr(_gipp, "shadow", False))
        for k in range(ctx_len, emb.size(1)):
            e_in = emb_hist[:, -ctx_len:]
            a_in = act_emb[:, k - ctx_len:k]
            a_raw = batch["action"][:, k - ctx_len:k]
            rollout_step = k - ctx_len + 1
            p = self.model.predict(
                e_in, a_in, action=a_raw, rollout_step=rollout_step,
                apply_gipp=not _shadow)[:, -1:]  # (B,1,D)
            preds.append(p)
            p_memory = _gipp(p, e_in, rollout_step=rollout_step) if _shadow else p
            emb_hist = torch.cat([emb_hist, p_memory], dim=1)
        pred_emb = torch.cat(preds, dim=1)   # (B, H, D), H = T - ctx_len
        tgt_emb = emb[:, ctx_len:]           # (B, H, D)
    else:
        tgt_emb = emb[:, n_preds:] # label
        pred_emb = self.model.predict(ctx_emb, ctx_act, action=ctx_action) # pred

    # LeWM loss. Optional pos_weight up-weights the physical slot dims in the
    # prediction loss -> forces the predictor/encoder to make the position slot
    # load-bearing (linear in true position), so the physics-governed slot carries
    # the prediction instead of the diluted 190-D black-box channel.
    _sq = (pred_emb - tgt_emb).pow(2)  # (B, H, D)
    _pw = float(cfg.loss.get("pos_weight", 1.0))
    _scfg = cfg.loss.get("structured", None)
    if _pw != 1.0 and _scfg is not None and float(_scfg.get("weight", 0.0)) > 0:
        _tc = _scfg.get("target", "proprio")
        _tc = [_tc] if isinstance(_tc, str) else list(_tc)
        _sd = int(_scfg.get("start_dim", 0))
        _pd = sum(int(batch[c].shape[-1]) for c in _tc)
        _w = torch.ones(_sq.size(-1), device=_sq.device)
        _w[_sd:_sd + _pd] = _pw
        output["pred_loss"] = (_sq * _w).mean()
    else:
        output["pred_loss"] = _sq.mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Structured latent slot loss (PIWM-style intrinsic physical slots).
    # This is stronger than probe loss: fixed embedding dimensions themselves
    # are supervised to equal physical quantities, with no readout head.
    structured_cfg = cfg.loss.get("structured", None)
    if structured_cfg is not None and float(structured_cfg.get("weight", 0.0)) > 0:
        tgt_cols = structured_cfg.get("target", "proprio")
        tgt_cols = [tgt_cols] if isinstance(tgt_cols, str) else list(tgt_cols)
        target = torch.cat([torch.nan_to_num(batch[c], 0.0) for c in tgt_cols], dim=-1)  # (B,T,P)
        # CONTROL (2026-07-16): shuffle the target across the batch -> the slot is
        # pinned to a same-distribution but PHYSICALLY MEANINGLESS target (another
        # trial's position). If pos_weight's OOD gain survives this, the gain is
        # capacity restriction / regularization, NOT physics knowledge.
        if bool(structured_cfg.get("shuffle_control", False)):
            target = target[torch.randperm(target.size(0), device=target.device)]
        start_dim = int(structured_cfg.get("start_dim", 0))
        end_dim = start_dim + target.size(-1)
        if start_dim < 0 or end_dim > emb.size(-1):
            raise ValueError(
                f"structured latent slot [{start_dim}:{end_dim}] does not fit emb dim {emb.size(-1)}"
            )
        slot = emb[..., start_dim:end_dim]
        output["structured_loss"] = (slot - target).pow(2).mean()
        output["loss"] = output["loss"] + structured_cfg.get("weight", 1.0) * output["structured_loss"]

    # Dynamics-consistency loss (NEW): constrain how the physical slot EVOLVES,
    # not just its value. The PREDICTED rollout slot's velocity (finite-diff
    # displacement) must match the true proprio velocity over the horizon. Unlike
    # structured/probe (which pin "what the state IS"), this constrains "how the
    # state CHANGES" -> directly targets long-horizon drift. Soft version of a
    # physics-equation constraint (no fixed accel form -> avoids the smooth-accel
    # head failing on collision impulses). accel_weight>0 adds 2nd-order
    # (acceleration) consistency. Designed for free_rollout; needs the physical
    # slot to be meaningful (use with structured.weight>0).
    consistency_cfg = cfg.loss.get("consistency", None)
    if consistency_cfg is not None and float(consistency_cfg.get("weight", 0.0)) > 0:
        c_cols = consistency_cfg.get("target", "proprio")
        c_cols = [c_cols] if isinstance(c_cols, str) else list(c_cols)
        c_sd = int(consistency_cfg.get("start_dim", 0))
        proprio_full = torch.cat([torch.nan_to_num(batch[c], 0.0) for c in c_cols], dim=-1)  # (B,T,P)
        c_ed = c_sd + proprio_full.size(-1)
        h_start = ctx_len if cfg.wm.get("free_rollout", False) else n_preds
        true_pos = proprio_full[:, h_start:]         # (B, H, P) aligned to pred horizon
        pred_slot = pred_emb[..., c_sd:c_ed]         # (B, H, P)
        H = min(pred_slot.size(1), true_pos.size(1))
        pred_slot, true_pos = pred_slot[:, :H], true_pos[:, :H]
        pred_vel = pred_slot[:, 1:] - pred_slot[:, :-1]
        true_vel = true_pos[:, 1:] - true_pos[:, :-1]
        cons = (pred_vel - true_vel).pow(2).mean()
        aw = float(consistency_cfg.get("accel_weight", 0.0))
        if aw > 0 and H >= 3:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            true_acc = true_vel[:, 1:] - true_vel[:, :-1]
            cons = cons + aw * (pred_acc - true_acc).pow(2).mean()
        output["consistency_loss"] = cons
        output["loss"] = output["loss"] + float(consistency_cfg["weight"]) * output["consistency_loss"]

    # Accel regularization (keeps the learnable acceleration near 0). uniform_motion's
    # true acceleration is 0; a free accel MLP overfits ID-specific residuals and wrecks
    # v-OOD / long-horizon. Penalizing ||accel||^2 pulls it back toward constant-velocity.
    _dyn_cfg_f = cfg.get("dynamics", None)
    if (_dyn_cfg_f is not None and float(_dyn_cfg_f.get("accel_reg", 0.0)) > 0
            and getattr(self.model, "dynamics", None) is not None
            and getattr(self.model.dynamics, "last_accel_sq", None) is not None):
        output["accel_reg_loss"] = self.model.dynamics.last_accel_sq
        output["loss"] = output["loss"] + float(_dyn_cfg_f.get("accel_reg")) * output["accel_reg_loss"]

    # Deep-supervision linear probe loss (PIWM-style, arXiv:2504.03861).
    # Aligns projector-space emb with physical state so emb carries linearly-
    # decodable physics -> rollout-predicted embs decode better + reduced drift.
    # target can be a single column ("proprio") or a list (["proprio","action"])
    # to supervise position + velocity jointly. Toggle via loss.probe.enabled.
    # Probe loss is active iff its weight > 0 (no separate enabled flag — weight=0 == off,
    # which makes weight the single knob and the natural baseline arm of a λ sweep).
    probe_cfg = cfg.loss.get("probe", None)
    if probe_cfg is not None and float(probe_cfg.get("weight", 0.0)) > 0:
        tgt_cols = probe_cfg.get("target", "proprio")
        tgt_cols = [tgt_cols] if isinstance(tgt_cols, str) else list(tgt_cols)
        kw = int(probe_cfg.get("frames", 1))
        if kw <= 1:
            # single-frame: probe each frame's emb -> its own physical state
            target = torch.cat([torch.nan_to_num(batch[c], 0.0) for c in tgt_cols], dim=-1)  # (B,T,sumP)
            probe_pred = self.model.probe_head(emb)  # (B,T,sumP)
        else:
            # multi-frame: stack first kw frame embs (window) -> predict last frame's state.
            # Lets the probe do cross-frame differencing -> velocity becomes decodable,
            # and relaxes the per-frame position rigidity that hurt vx on OOD.
            kw = min(kw, emb.size(1))
            win = emb[:, :kw].flatten(1)  # (B, kw*D)
            target = torch.cat([torch.nan_to_num(batch[c][:, kw - 1], 0.0) for c in tgt_cols], dim=-1)  # (B,sumP)
            probe_pred = self.model.probe_head(win)  # (B,sumP)
        output["probe_loss"] = (probe_pred - target).pow(2).mean()
        output["loss"] = output["loss"] + probe_cfg.get("weight", 1.0) * output["probe_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    _predictor_use_action = bool(cfg.wm.get("use_action", True))
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            # A passive dataset still carries a schema-mandated action column.
            # When the predictor is action-free, leave that constant placeholde
            # unnormalized (zero variance is expected, not an error).
            if col == "action" and not _predictor_use_action:
                setattr(cfg.wm, "action_dim", dataset.get_dim(col))
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    # encoder_type=dinov2: DINO-WM / V-JEPA2-AC style JEPA variant — frozen
    # pretrained SSL backbone (DINOv2-small, 384-d CLS), only the projector
    # (384->192, acts as trainable adapter) + predictor stack train. Set
    # freeze_encoder=true with this. Everything downstream (losses, FR/TF,
    # injection arms, eval) is unchanged -> controlled cross-backbone ablation.
    if cfg.get("encoder_type", "vit_hf") == "dinov2":
        from transformers import Dinov2Model
        _d2 = cfg.get("dinov2_path",
                      "/data1/likun-share/junjxu/.stable_worldmodel/hf_dinov2_small")
        encoder = Dinov2Model.from_pretrained(_d2)
        print(f"[encoder_type=dinov2] frozen DINOv2 backbone from {_d2} "
              f"(hidden={encoder.config.hidden_size})", flush=True)
    else:
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )
    world_model.use_action = bool(cfg.wm.get("use_action", True))
    print(f"[protocol] predictor use_action={world_model.use_action}", flush=True)

    _gipp_cfg = cfg.get("gipp", None)
    if _gipp_cfg is not None and bool(_gipp_cfg.get("enabled", False)):
        _state_path = _gipp_cfg.get("state_path", None)
        if not _state_path:
            raise ValueError("gipp.enabled=true requires gipp.state_path")
        world_model.gipp = GaugeInvariantPhysicsProjection.from_npz(
            _state_path,
            alpha=float(_gipp_cfg.get("alpha", 1.0)),
            eps=float(_gipp_cfg.get("eps", 1e-4)),
            gate=str(_gipp_cfg.get("gate", "constant")),
            gate_threshold=float(_gipp_cfg.get("gate_threshold", 0.15)),
            gate_temperature=float(_gipp_cfg.get("gate_temperature", 0.05)),
            physics=str(_gipp_cfg.get("physics", "constant_velocity")),
            gravity=_gipp_cfg.get("gravity", None),
            horizon_start=float(_gipp_cfg.get("horizon_start", 8.0)),
            horizon_temperature=float(_gipp_cfg.get("horizon_temperature", 2.0)),
            damping=float(_gipp_cfg.get("damping", 0.95)),
            acceleration_clip=float(_gipp_cfg.get("acceleration_clip", 3.0)),
            collision_distance=float(_gipp_cfg.get("collision_distance", 2.5)),
        )
        world_model.gipp.shadow = bool(_gipp_cfg.get("shadow", False))
        print(f"[gipp] loaded frozen projection from {_state_path}; "
              f"alpha={world_model.gipp.alpha} physics={world_model.gipp.physics} "
              f"shadow={world_model.gipp.shadow}", flush=True)
        if bool(_gipp_cfg.get("freeze_projector", True)):
            for p in world_model.projector.parameters():
                p.requires_grad_(False)
            world_model.projector.eval()
            print("[gipp] froze representation projector to preserve decoder coordinates", flush=True)

    # Deep-supervision linear probe head (PIWM-style). Built unconditionally for
    # ckpt consistency; only contributes to loss when loss.probe.enabled=true.
    # Linear (not MLP) to match the deep-supervision paper (arXiv:2504.03861).
    # target may be a single column or a list -> output dim = sum of column dims.
    _probe_cfg = cfg.loss.get("probe", {})
    _probe_tgt = _probe_cfg.get("target", "proprio")
    _probe_tgt = [_probe_tgt] if isinstance(_probe_tgt, str) else list(_probe_tgt)
    probe_target_dim = sum(cfg.wm.get(f"{c}_dim", embed_dim) for c in _probe_tgt)
    _probe_frames = max(1, int(_probe_cfg.get("frames", 1)))  # multi-frame window for velocity decodability
    world_model.probe_head = torch.nn.Linear(embed_dim * _probe_frames, probe_target_dim)

    # Optional structured 2nd-order dynamics on the physical slot (PIWM-style).
    # Off by default; when enabled, the position slot evolves by a fixed kinematic
    # rule instead of the black-box predictor. See module.SecondOrderDynamics.
    _dyn_cfg = cfg.get("dynamics", None)
    if _dyn_cfg is not None and bool(_dyn_cfg.get("enabled", False)):
        # pos_dim: auto from the structured target columns (so the physical slot
        # width tracks the data -> uniform=2, collision=4 with zero code change).
        # Override by setting dynamics.pos_dim to an int in the config.
        _pd = _dyn_cfg.get("pos_dim", None)
        if _pd is None or _pd == "auto":
            _struct_tgt = cfg.loss.get("structured", {}).get("target", "proprio")
            _struct_tgt = [_struct_tgt] if isinstance(_struct_tgt, str) else list(_struct_tgt)
            _pd = sum(int(cfg.wm.get(f"{c}_dim")) for c in _struct_tgt)
        _use_action = bool(_dyn_cfg.get("use_action", False))
        _learn_accel = bool(_dyn_cfg.get("learnable_accel", True))
        _accel_form = str(_dyn_cfg.get("accel_form", "mlp"))
        world_model.dynamics = SecondOrderDynamics(
            pos_dim=int(_pd),
            act_dim=effective_act_dim if _use_action else 0,
            hidden=int(_dyn_cfg.get("hidden", 64)),
            use_action=_use_action,
            learnable_accel=_learn_accel,
            accel_form=_accel_form,
        )
        print(f"[dynamics] SecondOrderDynamics enabled: pos_dim={_pd} use_action={_use_action} "
              f"act_dim={effective_act_dim if _use_action else 0} learnable_accel={_learn_accel} "
              f"accel_form={_accel_form}", flush=True)

    init_ckpt = cfg.get("init_from_ckpt")
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ck, torch.nn.Module):
            ck = ck.state_dict()
        if isinstance(ck, dict) and "state_dict" in ck:
            ck = ck["state_dict"]
        # Transformers has used both ViT naming layouts across releases.  Score
        # the raw and remapped checkpoint keys against the actual instantiated
        # model and select the compatible representation instead of assuming a
        # package version.
        _raw_ck = ck
        _mapped_ck = _remap_old_vit_keys(ck)
        _target_keys = set(world_model.state_dict())
        _raw_score = sum(k in _target_keys for k in _raw_ck)
        _mapped_score = sum(k in _target_keys for k in _mapped_ck)
        ck = _mapped_ck if _mapped_score > _raw_score else _raw_ck
        print(f"[init_from_ckpt] key_match raw={_raw_score} remapped={_mapped_score}", flush=True)
        prefixes = tuple(cfg.get("init_load_prefixes", ["encoder.", "projector.", "pred_proj."]))
        filtered = {k: v for k, v in ck.items() if k.startswith(prefixes)}
        missing, unexpected = world_model.load_state_dict(filtered, strict=False)
        loaded = len(filtered) - len(unexpected)
        print(f"[init_from_ckpt] {init_ckpt}", flush=True)
        print(f"[init_from_ckpt] prefixes={prefixes} loaded={loaded} unexpected={len(unexpected)} missing={len(missing)}", flush=True)
        if unexpected[:3]:
            print(f"[init_from_ckpt] first unexpected: {unexpected[:3]}", flush=True)
        # Guard against the silent naming-drift bug: a pusht ViT ckpt has 198
        # encoder.* params; if almost none load the encoder is left random-init.
        n_enc = sum(1 for k in filtered if k.startswith("encoder."))
        n_enc_loaded = n_enc - sum(1 for k in unexpected if k.startswith("encoder."))
        if n_enc and n_enc_loaded < 0.5 * n_enc:
            raise RuntimeError(
                f"[init_from_ckpt] only {n_enc_loaded}/{n_enc} encoder weights loaded "
                f"from {init_ckpt} — key naming mismatch, encoder would be random-init. "
                f"Update _remap_old_vit_keys()."
            )

    # Optional: freeze the encoder (keep pusht weights fixed) and train only the
    # predictor / projector / probe. Isolates "does finetuning the encoder help?"
    if cfg.get("freeze_encoder", False):
        n_frozen = 0
        for p in world_model.encoder.parameters():
            p.requires_grad_(False)
            n_frozen += 1
        world_model.encoder.eval()
        print(f"[freeze_encoder] froze {n_frozen} encoder param tensors (encoder kept at init)", flush=True)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    # Collect loggers; wandb (cloud) and tensorboard (local) can coexist.
    # An empty list -> logger=False, which disables Lightning's default CSVLogger
    # (flaky "dict contains fields not in fieldnames" crash when the logged
    # metric-key set changes between rows, e.g. hit collision FT at epoch-0 val).
    loggers = []
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(**cfg.wandb.config)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
        loggers.append(wandb_logger)
    tb_cfg = cfg.get("tensorboard", None)
    if tb_cfg is not None and tb_cfg.get("enabled", False):
        # Local, offline board. Logs -> <save_dir or run_dir>/<name>/.
        # No log_hyperparams: TB hparams only takes flat scalars and chokes on
        # the nested OmegaConf config; scalar metrics (losses) log automatically.
        loggers.append(
            TensorBoardLogger(
                save_dir=str(tb_cfg.get("save_dir", None) or run_dir),
                name=tb_cfg.get("name", "tb"),
                version="",
            )
        )
    logger = loggers or False

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=cfg.get("ckpt_every", 1),
        keep_last_n=cfg.get("ckpt_keep_last_n", 0),
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=cfg.get("enable_lightning_ckpt", True),
    )

    _resume_ckpt = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=_resume_ckpt if _resume_ckpt.is_file() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
