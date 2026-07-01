"""Pixel decoder for visualizing what a (frozen) LeWM embedding encodes.

The LeWM world model consumes a single 192-d CLS embedding per frame
(`encoder(x).last_hidden_state[:, 0]`, optionally `projector`-ed). That vector
is what the predictor rolls out, so "what physics does the latent carry?" is
really "what can be recovered from that 192-d vector?".

This module trains a small upsampling decoder  z (192) -> image (3,224,224)
against a FROZEN encoder. Because the encoder is frozen, every frame's
embedding is constant, so we precompute embeddings once and train only the
decoder. The reconstruction is a direct, visual read-out of the embedding's
content: if the ball reappears at the right place, position is encoded.

NOTE: a frozen-encoder feature-inversion decoder shows what is *linearly+
nonlinearly recoverable* from the embedding. A blurry / mislocated ball means
the embedding dropped that information; a crisp one means it kept it.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn


# ImageNet normalization used by the LeWM image preprocessor (utils.get_img_preprocessor)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def remap_old_vit_keys(sd):
    """Copy of train._remap_old_vit_keys — old HF ViT names -> new transformers
    naming, so the pusht weights.pt encoder body actually loads (idempotent)."""
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


class UpBlock(nn.Module):
    """Bilinear upsample + 3x3 conv (avoids ConvTranspose checkerboard artifacts)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(self.up(x))))


class LatentDecoder(nn.Module):
    """z (B, emb_dim) -> image (B, 3, 224, 224) in [0, 1].

    7 -> 14 -> 28 -> 56 -> 112 -> 224 spatial, 5 upsampling stages.
    The decoder also standardizes the input embedding using stored
    (mean, std) so training is stable regardless of the encoder's feature scale.
    """

    def __init__(self, emb_dim=192, base=512, out_hw=224):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_hw = out_hw
        # standardization buffers (filled from training-set stats)
        self.register_buffer("emb_mean", torch.zeros(emb_dim))
        self.register_buffer("emb_std", torch.ones(emb_dim))

        self.fc = nn.Linear(emb_dim, base * 7 * 7)
        self.base = base
        self.up = nn.Sequential(
            UpBlock(base, 256),   # 7  -> 14
            UpBlock(256, 128),    # 14 -> 28
            UpBlock(128, 64),     # 28 -> 56
            UpBlock(64, 32),      # 56 -> 112
            UpBlock(32, 16),      # 112 -> 224
        )
        self.head = nn.Conv2d(16, 3, 3, padding=1)

    def set_norm(self, mean, std):
        self.emb_mean.copy_(mean.to(self.emb_mean))
        self.emb_std.copy_(std.clamp_min(1e-6).to(self.emb_std))

    def forward(self, z):
        z = (z - self.emb_mean) / self.emb_std
        x = self.fc(z).view(-1, self.base, 7, 7)
        x = self.up(x)
        return torch.sigmoid(self.head(x))


def build_frozen_encoder(ckpt_path, device="cuda", encoder_scale="tiny",
                         patch_size=14, img_size=224, emb_source="cls"):
    """Return (encode_fn, emb_dim).

    ckpt_path may be:
      - a state_dict .pt (e.g. lewm_paper_pusht/weights.pt = un-finetuned encoder)
      - a full JEPA object .ckpt (a finetuned model)

    encode_fn: (pix_chw_uint8 on device) -> (B, emb_dim) embedding, no_grad.
    emb_source: 'cls' = encoder CLS token (raw visual feature, default)
                'proj'= projector(CLS) = the latent the world model rolls out
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import stable_pretraining as spt

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    projector = None
    if isinstance(obj, nn.Module) and hasattr(obj, "encoder"):
        # full JEPA pickle (finetuned)
        encoder = obj.encoder
        if emb_source == "proj":
            projector = getattr(obj, "projector", None)
    else:
        # state_dict — build a fresh ViT and load the encoder (and projector) weights
        encoder = spt.backbone.utils.vit_hf(
            encoder_scale, patch_size=patch_size, image_size=img_size,
            pretrained=False, use_mask_token=False,
        )
        sd = obj["state_dict"] if (isinstance(obj, dict) and "state_dict" in obj) else obj
        sd = remap_old_vit_keys(sd)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        n_loaded = len(enc_sd) - len(unexpected)
        if n_loaded < 0.5 * max(1, len(enc_sd)):
            raise RuntimeError(
                f"only {n_loaded}/{len(enc_sd)} encoder weights loaded from {ckpt_path} "
                f"— naming mismatch, encoder would be random-init."
            )
        print(f"[encoder] loaded {n_loaded}/{len(enc_sd)} encoder weights, "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if emb_source == "proj":
            raise ValueError("emb_source='proj' needs a full JEPA object ckpt, not a bare state_dict.")

    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    if projector is not None:
        projector.eval().to(device)
        for p in projector.parameters():
            p.requires_grad_(False)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_fn(pix_chw_uint8):
        x = pix_chw_uint8.to(device).float() / 255.0
        x = (x - mean) / std
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        if projector is not None:
            h = projector(h)
        return h.float()

    # probe emb_dim with a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.uint8, device=device)
        emb_dim = encode_fn(dummy).shape[1]
    return encode_fn, emb_dim
