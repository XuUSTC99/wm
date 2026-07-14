
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

---

## Fork Extensions: Physical Structure & Training Protocol

This fork extends upstream LeWM with physical-structure injection and training-protocol changes, used for our PhyWorld / Physion studies. Everything is switched off by default (except free-rollout) and controlled from [`config/train/lewm.yaml`](config/train/lewm.yaml). Full experimental findings live in `../reports/6-24/final/` and `../reports/7-11/`.

### 1. Structured physical slot — how physical quantities are encoded

We designate the **first `P` dims of the 192-D latent as a physical slot** (`P` = proprio dim, e.g. 2 for ball position). Nothing is hard-written into the embedding: the encoder still computes all 192 dims from pixels, and a **soft MSE loss** pulls the slot toward the physical state ([`train.py`](train.py), "Structured latent slot loss"):

```python
slot = emb[..., 0:P]                       # encoder output, all T frames
structured_loss = (slot - target).pow(2).mean()
loss += cfg.loss.structured.weight * structured_loss
```

Two details that matter:

- **The target is z-score-normalized proprio, not raw coordinates.** Each column is normalized to ~N(0,1) by dataset statistics (`utils.get_column_normalizer`), matching the Gaussian latent scale enforced by SIGReg. Pinning raw world coordinates (range ~1–12) onto a ~N(0,1) latent dim would fight the regularizer.
- **It is training-time supervision only.** At inference the encoder produces the slot from pixels alone; after convergence `structured_loss ≈ 0.015`, i.e. the slot ≈ normalized position to ~0.1σ. Unlike a probe (`probe_head(emb) ≈ state`, readout head, weak constraint), the structured slot has *no head* — the latent dims themselves are the state (hard assignment).

```yaml
loss:
  structured:
    weight: 1.0        # 0 = off
    target: proprio    # column(s); slot width = summed dim
    start_dim: 0
```

### 2. Kinematics dynamics head — how the equation is wired in

`dynamics.enabled=true` attaches a `SecondOrderDynamics` module ([`module.py`](module.py)) that **surgically replaces the predictor's output on the slot dims only**. The black-box transformer still predicts all 192 dims; we then discard its first `P` dims and substitute a kinematic extrapolation computed from the *input* embedding's slot ([`jepa.py`](jepa.py) `predict`):

```python
preds = self.pred_proj(self.predictor(emb, act_emb))       # black-box, 192-D
pos_next = self.dynamics(emb[..., :P], action)             # kinematics on the slot
preds = torch.cat([pos_next, preds[..., P:]], dim=-1)      # physics P dims + black-box rest
```

The module is discrete second-order kinematics (Δt=1 absorbed into units):

```
v_t      = p_t − p_{t−1}          # finite-difference velocity from two frames
p_{t+1}  = p_t + v_t + a_t        # == 2·p_t − p_{t−1} + a_t
```

with the acceleration term selectable via `dynamics.accel_form`:

| `accel_form` | a_t | notes |
|---|---|---|
| `none` | 0 | pure constant-velocity extrapolation |
| `const` | learnable constant `g` (one vector) | **strict PIWM-style**: fixed known form, only the physical parameter (gravity) is learned |
| `mlp` | zero-init MLP(p, v[, action]) | free-form correction (non-PIWM baseline; overfits) |

All forms start at exact constant velocity (zero-init). The head participates in both the training forward pass and autoregressive `rollout()` (both route through `predict`), so train/eval behavior is consistent. It requires `loss.structured.weight > 0` — the equation only makes sense if `emb[..., :P]` actually encodes position.

```yaml
dynamics:
  enabled: false
  pos_dim: null        # null = auto from structured target dim (uniform=2, collision=4)
  accel_form: mlp      # none | const | mlp
  accel_reg: 0.0       # optional ||a||² penalty (mlp form)
```

### 3. Training protocol & other switches

- **`wm.free_rollout: true` (default in this fork).** Upstream LeWM trains single-step teacher-forced; we unroll the predictor autoregressively for `wm.num_preds` steps, feeding its own predictions back, and supervise every step. This closes the train/test exposure-bias gap and is the single largest, most portable gain we found (all three PhyWorld domains + real Physion++). Set `wm.free_rollout=false wm.num_preds=1` to recover upstream behavior.
- **`wm.num_preds`**: match rollout length to dynamics complexity (collision ≈ 20; smooth domains 8 — longer hurts there).
- **`loss.pos_weight`**: up-weights the slot dims inside the prediction loss (makes the slot "load-bearing").
- **`loss.consistency.*`**: form-free velocity/acceleration consistency on *predicted* slots.
- **`aug.appearance / aug.scale / aug.temporal`**: per-trial photometric jitter / center-zoom / frame-stride augmentation (pure-video, label-free).
- **`loss.probe.*`**: deep-supervision readout probe (kept from earlier experiments).

### 4. What we found (honest summary)

Across PhyWorld (uniform / parabola / collision) and Physion/Physion++: **all physical-structure variants — structured slot, kinematics head in every accel form (incl. strict PIWM `const`), probe, consistency, label-free, from-scratch or fine-tuned — failed to improve OOD/long-horizon rollout and usually hurt.** Root cause is architectural (a 2/192 slot in a shared latent is not load-bearing; the black-box channel redundantly encodes position and the predictor routes around the slot), not the equations. What robustly helps: **free-rollout training, horizon matched to domain complexity, and domain-matched augmentation** (whose gains do *not* transfer from synthetic to real data). See `../reports/6-24/final/FINAL_SUMMARY.md` for the complete ledger.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
