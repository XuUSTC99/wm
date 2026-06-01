# wm — World Model experiments

> **Path note (2026-05)**: this tree was renamed `agent_memory/` → `am/`. All paths are now `/home/qlib/am/wm/...`. A symlink `~/lewm_run -> /home/qlib/am/wm/le-wm` exists for launching training.

Three related projects sit side by side here:

| Subdir | What it is | Upstream |
|---|---|---|
| [`le-wm/`](./le-wm/) | **LeWorldModel**: JEPA-style action-conditioned world model trained from pixels. Includes training, planning, evaluation. | https://github.com/lucas-maes/le-wm |
| [`phyworld/`](./phyworld/) | **How Far is Video Generation from World Model**: physical-law benchmark with code to generate / evaluate ID & OOD video data (uniform motion, collision, parabola). | https://github.com/PhyWorld/PhyWorld |
| [`PIWM/`](./PIWM/) | **Physically Interpretable World Models**: latent↔physics alignment + physics-structured dynamics. Design source for the deep-supervision experiments. | arXiv:2412.12870 / 2503.02143 |

The reason they're together: phyworld's data tests whether a video / world model has actually learned physics. We feed phyworld trajectories into `le-wm` and probe how the JEPA encoder + predictor behave on this benchmark; PIWM supplies ideas for *improving* that behavior.

For project-internal docs see [`le-wm/README.md`](./le-wm/README.md) and [`phyworld/README.md`](./phyworld/README.md). This README covers the **bridge between them** + an index of the experiment reports.

---

## Experiment reports (latest → oldest)

All under [`reports/`](./reports/). Read [`reports/5-26/negtive_result_report.md`](./reports/5-26/negtive_result_report.md) first — it's the current source of truth.

| Report | Topic | Headline finding |
|---|---|---|
| [5-26/negtive_result_report.md](./reports/5-26/negtive_result_report.md) | Main report: probe protocol + metric fixes; frozen vs FT across 3 domains | "OOD encoder collapse" was a R²+K=1+ID-only-fit artifact. With MSE+ρ+K=4 MLP, representations are consistent across all partitions. |
| [5-26/§6.4](./reports/5-26/negtive_result_report.md) | **ID-only FT** (leak-free): LeWM + DiT LoRA on official `*_30K` ID data, probe on full OOD eval | True ID→OOD FT gain ≈ 0 (LeWM) / large net-negative (DiT). Earlier "+0.02 ρ" was partition-memorization, not generalization. |
| [5-26/rollout_results.md](./reports/5-26/rollout_results.md) | **AR rollout** (uses ARPredictor, not just encoder) | 1-step prediction great (cos 0.98-0.99); multi-step AR drifts (collision fastest). Encoding current state ≠ predicting trajectory. |
| [5-26/piwm_deepsup_results.md](./reports/5-26/piwm_deepsup_results.md) | **PIWM deep-supervision** linear probe in FT loss (parabola; 4 arms) | Single-frame probe helps position/vy/long-cos but damages vx on high-speed OOD. **Training-time multi-frame probe (`frames=4`) is the fix** — recovers vx + best long-horizon cos. |
| [5-26/arpredictor_rollout_proposal.md](./reports/5-26/arpredictor_rollout_proposal.md) | Proposal doc behind the rollout experiment | — |
| [5-19/](./reports/5-19/) , [5-12/](./reports/5-12/) | Earlier DiT / collision / uniform-motion reports (R²-era, partly superseded) | — |

---

## Environment

A single `uv` venv lives in [`le-wm/.venv/`](./le-wm/) (Python 3.10, `torch==2.9.1+cu128`, `stable-worldmodel[train,env]`, plus `imageio`, `Pillow`, `h5py` already pulled in transitively). All commands below use it.

```bash
cd /home/qlib/am/wm/le-wm
source .venv/bin/activate
```

If the venv ever needs to be rebuilt:

```bash
cd /home/qlib/am/wm/le-wm
uv venv --python=3.10
source .venv/bin/activate
uv pip install swig                       # box2d-py needs the swig binary at build time
cat > /tmp/lewm-build-constraints.txt <<'EOF'
setuptools<=66.0.0
wheel<=0.38.4
pip<=23.0.1
EOF
uv pip install --build-constraints /tmp/lewm-build-constraints.txt 'stable-worldmodel[train,env]'
uv pip install -U 'datasets>=2.14,<4'     # upstream pin is too old (1.1.1)
# system has CUDA 12.9 driver, swap default cu130 wheel for cu128:
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128'
```

---

## Datasets

Datasets live at `$STABLEWM_HOME` (defaults to `~/.stable_worldmodel/`). `stable_worldmodel.HDF5Dataset` looks up `<name>.h5` there.

| File | Source | Size | Notes |
|---|---|---|---|
| `pusht_expert_train.h5` | LeRobot pusht (HF: `quentinll/lewm-pusht`) | 44 GB | Reference dataset for `le-wm`. 2.3M frames, 18685 episodes, action+proprio+state. |
| `phyworld_{uniform_motion,parabola}.h5`, `phyworld_collision_eval.h5` | converters below | ~100 MB | **Eval sets** (all 4 partitions: ID + r/m-OOD + v-OOD + both-OOD). Used for probing / rollout. |
| `phyworld_{collision,uniform_motion,parabola}_id1k.h5` | converters, `--limit 1000` on official `*_30K.hdf5` | ~100-160 MB | **ID-only training sets** (1000 traj, 32k frames, 100% ID). For leak-free ID→OOD FT. Built from HF `magicr/phyworld` `id_ood_data/*_30K.hdf5`. |

Converters: [`convert_to_lewm.py`](./phyworld/scripts/convert_to_lewm.py) (uniform/parabola, action=velocity), [`convert_collision_to_lewm.py`](./phyworld/scripts/convert_collision_to_lewm.py) (collision, action=acceleration). **Action semantics differ per domain** — uniform/parabola use velocity, collision uses acceleration; the LeWM `action_encoder` dim must match (2 vs 4).

---

## phyworld → le-wm bridge

### What [`convert_to_lewm.py`](./phyworld/scripts/convert_to_lewm.py) does

phyworld's hdf5 stores each trajectory as an MP4 byte-blob in `video_streams/<group>/<idx>` and 2D positions in `position_streams/...`. It has **no action signal** — it's a passive physics dataset.

le-wm wants a single flat-stacked hdf5 with `pixels`, `action`, `proprio`, `ep_len`, `ep_offset`, `episode_idx`, `step_idx`.

The script:

1. Decodes each trajectory's MP4 → `(T, H, W, 3)` uint8.
2. Resizes to `--img-size` (default 224, matches le-wm's `img_size`).
3. Uses `position[t]` directly as `proprio[t]`.
4. **Synthesises** `action[t] = position[t+1] - position[t]` (velocity). Reason: le-wm's column normalizer divides by `std`; an all-zero `action` would produce NaNs. Velocity is also a physically sensible "control" for uniform motion.
5. Writes ep bookkeeping (`ep_len = 32` per traj, etc.).

Run from the `phyworld/` dir (so default `--src` resolves to `data/uniform_motion_eval.hdf5`):

```bash
cd /home/qlib/am/wm/phyworld
python scripts/convert_to_lewm.py
# -> writes ~/.stable_worldmodel/phyworld_uniform_motion.h5  (~100 MB, 2-3 min)
```

Useful flags:

```bash
python scripts/convert_to_lewm.py --limit 4 --dst /tmp/phyworld_test.h5  # quick check
python scripts/convert_to_lewm.py --img-size 256 --name phyworld_um256   # custom output
python scripts/convert_to_lewm.py --src path/to/other_phyworld.hdf5      # different source
```

### le-wm config for phyworld

Already created at [`le-wm/config/train/data/phyworld.yaml`](./le-wm/config/train/data/phyworld.yaml):

```yaml
dataset:
  name: phyworld_uniform_motion          # → ~/.stable_worldmodel/phyworld_uniform_motion.h5
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 1                           # phyworld traj is only 32 frames; no skip
  keys_to_load: [pixels, action, proprio]
  keys_to_cache: [action, proprio]
```

Differences vs `pusht.yaml`: `frameskip=1` (was 5) because trajectories are short; no `state` key (phyworld has no env truth state).

---

## Running training

### Smoke test — verifies the whole pipeline in ~10 s

```bash
cd /home/qlib/am/wm/le-wm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python train.py \
  data=phyworld \
  output_model_name=lewm_phyworld \
  wandb.enabled=False \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=1
```

Success looks like:

```
Trainer.fit stopped: `max_epochs=1` reached.
Epoch 0/0 ━━━━━━━ 2/2 0:00:03
fit/pred_loss = 0.24    fit/sigreg_loss = 40.0
```

For pusht swap `data=pusht` and drop the `output_model_name` override (or use a different name).

### Full training

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
  data=phyworld \
  output_model_name=lewm_phyworld \
  wandb.enabled=False
# checkpoints land in ~/.stable_worldmodel/lewm_phyworld_epoch_<N>_object.ckpt
```

To enable W&B, edit [`le-wm/config/train/lewm.yaml`](./le-wm/config/train/lewm.yaml) and set `wandb.config.entity / project`, then drop `wandb.enabled=False`.

---

## Deep-supervision probe (PIWM-style) — added in `lewm.yaml`

LeWM FT can add a linear probe loss that aligns the projector-space emb with physical state (PIWM principle 1). Off by default → baseline unchanged; toggle for ablation. Config block in [`le-wm/config/train/lewm.yaml`](./le-wm/config/train/lewm.yaml):

```yaml
loss:
  probe:
    enabled: false          # true = +probe arm
    weight: 1.0
    target: proprio         # single col, or list e.g. [proprio, action] (pos+vel)
    frames: 1               # 1 = single-frame probe; K>1 = stack K frame embs (velocity-decodable)
```

Key result ([reports/5-26/piwm_deepsup_results.md](./reports/5-26/piwm_deepsup_results.md)): **single-frame** probe (`frames=1`) helps position / vy / long-horizon cos but **damages vx on high-speed OOD** (single frame can't encode instantaneous velocity). **Multi-frame** (`frames=4`, `target=[proprio,action]`) is the fix — recovers vx and gives the best long-horizon rollout cosine. Example:

```bash
cd ~/lewm_run && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u train.py \
  data=phyworld_parabola_id1k loss.probe.enabled=true \
  'loss.probe.target=[proprio,action]' loss.probe.frames=4 \
  output_model_name=lewm_parabola_piwm_mf4_id1k subdir=parabola_piwm_mf4_id1k \
  trainer.max_epochs=20 +init_from_ckpt=~/.stable_worldmodel/lewm_paper_pusht/weights.pt
```

Eval / AR-rollout: [`phyworld/scripts/rollout_eval_id1k.py`](./phyworld/scripts/rollout_eval_id1k.py) (`--ckpt`/`--tag` to swap arms; reports latent cos vs horizon/partition + K=1/K=4 decoded pos/vel ρ).

---

## GPU selection

The host's 4× A6000s are usually shared. Pick the freest card:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv
```

A le-wm forward+backward at default batch size (128) needs ~15 GB. Set `CUDA_VISIBLE_DEVICES=<idx>` accordingly.

---

## Caveats / known issues

- **Auto-resume on collision**: le-wm auto-resumes from `~/.stable_worldmodel/<output_model_name>_weights.ckpt` if it exists. Switching `data=` between datasets changes the action_dim → state_dict shape mismatch. Use a distinct `output_model_name` per task (the example above does).
- **phyworld is not a real training set**: 36k frames of one-ball uniform motion is far too small + too simple for serious WM training; it's an *evaluation* set. For meaningful training generate larger sets via `phyworld/id_ood_data/*.py` (30k–3M videos) and rerun the converter pointing at the new hdf5.
- **action is synthetic** for phyworld. The model effectively learns near-unconditional next-frame prediction. Do not interpret loss curves as "le-wm learned physics from phyworld" — see the project root [`phyworld/README.md`](./phyworld/README.md) for the intended evaluation protocol.
