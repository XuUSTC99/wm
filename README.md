# wm — World Model experiments

Two related projects sit side by side here:

| Subdir | What it is | Upstream |
|---|---|---|
| [`le-wm/`](./le-wm/) | **LeWorldModel**: JEPA-style action-conditioned world model trained from pixels. Includes training, planning, evaluation. | https://github.com/lucas-maes/le-wm |
| [`phyworld/`](./phyworld/) | **How Far is Video Generation from World Model**: physical-law benchmark with code to generate / evaluate ID & OOD video data (uniform motion, collision, parabola). | https://github.com/PhyWorld/PhyWorld |

The reason they're together: phyworld's data tests whether a video / world model has actually learned physics. We want to feed phyworld trajectories into `le-wm` and see how the JEPA encoder + predictor behave on this benchmark.

For project-internal docs see [`le-wm/README.md`](./le-wm/README.md) and [`phyworld/README.md`](./phyworld/README.md). This README only covers the **bridge between them**.

---

## Environment

A single `uv` venv lives in [`le-wm/.venv/`](./le-wm/) (Python 3.10, `torch==2.9.1+cu128`, `stable-worldmodel[train,env]`, plus `imageio`, `Pillow`, `h5py` already pulled in transitively). All commands below use it.

```bash
cd /home/qlib/agent_memory/wm/le-wm
source .venv/bin/activate
```

If the venv ever needs to be rebuilt:

```bash
cd /home/qlib/agent_memory/wm/le-wm
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
| `phyworld_uniform_motion.h5` | Built by [`phyworld/scripts/convert_to_lewm.py`](./phyworld/scripts/convert_to_lewm.py) | 100 MB | 1152 traj × 32 frames = 36k frames. Eval-only set; no real action — synthesised from positions. |

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
cd /home/qlib/agent_memory/wm/phyworld
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
cd /home/qlib/agent_memory/wm/le-wm
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
