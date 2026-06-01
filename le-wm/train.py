import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Deep-supervision linear probe loss (PIWM-style, arXiv:2504.03861).
    # Aligns projector-space emb with physical state so emb carries linearly-
    # decodable physics -> rollout-predicted embs decode better + reduced drift.
    # target can be a single column ("proprio") or a list (["proprio","action"])
    # to supervise position + velocity jointly. Toggle via loss.probe.enabled.
    probe_cfg = cfg.loss.get("probe", None)
    if probe_cfg is not None and probe_cfg.get("enabled", False):
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
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
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

    init_ckpt = cfg.get("init_from_ckpt")
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            ck = ck["state_dict"]
        prefixes = tuple(cfg.get("init_load_prefixes", ["encoder.", "projector.", "pred_proj."]))
        filtered = {k: v for k, v in ck.items() if k.startswith(prefixes)}
        missing, unexpected = world_model.load_state_dict(filtered, strict=False)
        loaded = len(filtered) - len(unexpected)
        print(f"[init_from_ckpt] {init_ckpt}", flush=True)
        print(f"[init_from_ckpt] prefixes={prefixes} loaded={loaded} unexpected={len(unexpected)} missing={len(missing)}", flush=True)
        if unexpected[:3]:
            print(f"[init_from_ckpt] first unexpected: {unexpected[:3]}", flush=True)

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

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

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

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
