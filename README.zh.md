# wm — 物理世界模型实验

> 英文版见 [`README.md`](./README.md)。本文档是中文版项目介绍。
> **路径备注（2026-05）**：树根从 `agent_memory/` 改名为 `am/`，所有路径现在是 `~/am/wm/...`；A500 上对应 `~/junjxu/wm/...`。`~/lewm_run -> ~/am/wm/le-wm` 是启动训练的软链。

---

## 项目简介

把三个相关的子项目并排放在一起，互相喂数据 / 互相验证：

| 子目录 | 内容 | 上游 |
|---|---|---|
| [`le-wm/`](./le-wm/) | **LeWorldModel**：基于 JEPA 的 action-conditioned 像素世界模型（含训练 / 规划 / 评估）| https://github.com/lucas-maes/le-wm |
| [`phyworld/`](./phyworld/) | **How Far is Video Generation from World Model**：物理 benchmark（匀速 / 碰撞 / 抛物线），含 ID / OOD 视频生成与评估代码 | https://github.com/PhyWorld/PhyWorld |
| [`PIWM/`](./PIWM/) | **Physically Interpretable World Models**：latent ↔ 物理对齐 + 物理结构化动力学。本项目 deep-supervision 实验的设计参考 | arXiv:2412.12870 / 2503.02143 |

**为什么三个项目放在一起**：phyworld 的数据用来测一个视频/世界模型有没有真学到物理。我们把 phyworld 轨迹喂进 `le-wm`，探针 (probe) 测它的 JEPA encoder + predictor 在 ID/OOD 上的行为；PIWM 提供"如何改进 world model"的思想。

**Deep-supervision 实验的方法引用**：
- *Physically Interpretable World Models via Weakly Supervised Representation Learning* — arXiv:2412.12870（latent↔物理对齐；原则 1）
- *Four Principles for Physically Interpretable World Models* — arXiv:2503.02143（四个设计原则）
- *Improving World Models using Deep Supervision with Linear Probes* — arXiv:2504.03861（**我们实际实现的 recipe**：在 next-frame prediction loss 上加一个 linear probe 项，提升可解码性 + 减少 rollout 漂移）

子项目内部文档见 [`le-wm/README.md`](./le-wm/README.md) 和 [`phyworld/README.md`](./phyworld/README.md)。本 README 关注的是**三者间的桥接** + 实验报告索引。

---

## 实验报告（新 → 旧）

全部在 [`reports/`](./reports/) 下。新人接手按下面三个先后顺序看：

1. **真相源（qlib 原始）**：[`reports/5-26/negtive_result_report.md`](./reports/5-26/negtive_result_report.md) — 主要实证发现
2. **最新修复版重跑**：[`reports/6-2/piwm_three_domains_A800.md`](./reports/6-2/piwm_three_domains_A800.md) — 2026-06-08 A800 修好 init bug 后的重跑
3. **方法论 / 红线**：[`reports/6-24/diagnostic_report.md`](./reports/6-24/diagnostic_report.md) — 为什么 K=4 ρ / latent cos 会误导，以及怎么正确诊断 deep-sup

### ⚠️ A500 init bug 重要警告（2026-06-07）

2026-06-05 到 06-07 之间所有在 A500 上训练的 ckpt，`init_from_ckpt` 因 transformers 命名漂移**静默丢掉了 192/216 个 ViT 主体权重**（`encoder.encoder.layer.N.attention.attention.{q,k,v}` → 新版 `encoder.layers.N.attention.{q,k,v}_proj`）。

→ 这批 ckpt 的 encoder ≈ 在 1000 条轨迹上 FT 20ep 的**近随机网络**，**不是 pusht 预训练**。

下表里标记 "❌ broken init" 的报告**不可引用其数值**。修复方案已合入 [`le-wm/train.py`](./le-wm/train.py)（`_remap_old_vit_keys()` + 加载守卫）。今后任何训练 log 必须出现 `[init_from_ckpt] loaded=216 unexpected=0`，否则整批作废。

### 报告索引（按状态标注）

| 报告 | 日期 | 主题 | 状态 |
|---|---|---|---|
| [6-24/diagnostic_report.md](./reports/6-24/diagnostic_report.md) | 2026-06-24 | **诊断手册**：K=4 ρ 是 probe-loss 对偶 / pred_loss 是 ground truth / intrinsic dim 塌方 / encoder swap 测耦合 | ⚠️ 方法论可信，数值来自 broken-init ckpt（待重跑） |
| [6-2/piwm_three_domains_A800.md](./reports/6-2/piwm_three_domains_A800.md) | 2026-06-08 | **修复版重跑**：3 域 × 4 臂（baseline / pos-only / pos+vel / mf4），替代之前的 broken-init sweep | ✅ 有效（loaded=216） |
| [6-2/sweep_three_domains_results.md](./reports/6-2/sweep_three_domains_results.md) | 2026-06-06 | 45-config λ × frames sweep 三域 sweep（broken-init）| ❌ broken init；仅作踩坑记录 |
| [6-2/piwm_three_domains.md](./reports/6-2/piwm_three_domains.md) | 2026-06-02 | qlib 原始三域 deep-sup 对比 + within-traj std 单帧/多帧假说 | ✅ 有效（qlib 原始）|
| [6-2/piwm_uniform_collision_results.md](./reports/6-2/piwm_uniform_collision_results.md) | 2026-06-02 | qlib 原始 uniform + collision 数据表 | ✅ 有效（qlib 原始）|
| [6-2/idea-stage/IDEA_REPORT.md](./reports/6-2/idea-stage/IDEA_REPORT.md) | 2026-06-01 | Idea discovery — 下一步研究方向（PhysConsist-Rollout 等）| ✅ 规划文档 |
| [5-26/negtive_result_report.md](./reports/5-26/negtive_result_report.md) | 2026-05-26 | **主报告**：probe 协议 / 指标修正；pusht-only zero-shot ρ ≈ 0.9；ID-only FT 增益 ≈ 0 | ✅ 有效（qlib 原始）|
| [5-26/piwm_deepsup_results.md](./reports/5-26/piwm_deepsup_results.md) | 2026-05-26 | PIWM deep-sup linear probe；单帧伤 vx-OOD，`frames=4` 恢复 | ✅ 有效（qlib 原始）|
| [5-26/rollout_results.md](./reports/5-26/rollout_results.md) | 2026-05-26 | AR rollout（ARPredictor）：1 步极准，多步漂移，collision 最快 | ✅ 有效（qlib 原始）|
| [5-26/arpredictor_rollout_proposal.md](./reports/5-26/arpredictor_rollout_proposal.md) | 2026-05-26 | rollout 实验提案文档 | ✅ 有效（qlib 原始）|
| [5-19/FINAL_REPORT.md](./reports/5-19/FINAL_REPORT.md) | 2026-05-19 | LeWM 能学到牛顿运动定律吗？phyworld 完整实验 | ⚠️ 部分被取代（R² 时代指标）|
| [5-19/DIT_REPORT.md](./reports/5-19/DIT_REPORT.md) | 2026-05-19 | DiT-XL-2 在 phyworld collision（zero-shot + LoRA + LeWM pusht-only 对照）| ⚠️ 部分被取代 |
| [5-19/finetune_analyze.md](./reports/5-19/finetune_analyze.md) | 2026-05-19 | 为什么在 collision 上 SSL FT 反而让 probe 变差 | ⚠️ 部分被取代 |
| [5-12/COLLISION_REPORT.md](./reports/5-12/COLLISION_REPORT.md) | 2026-05-12 | LeWM 在 phyworld collision — 早期报告 | ⚠️ R² 时代，被 5-26 取代 |
| [5-12/UNIFORM_MOTION_REPORT.md](./reports/5-12/UNIFORM_MOTION_REPORT.md) | 2026-05-12 | LeWM 在 phyworld uniform — 早期报告 | ⚠️ R² 时代，被 5-26 取代 |
| [5-12/SLIDES.md](./reports/5-12/SLIDES.md) | 2026-05-12 | 早期发现的幻灯片 | — |

---

## 环境

单一 `uv` venv 在 [`le-wm/.venv/`](./le-wm/) 下（Python 3.10，`torch==2.9.1+cu128`，`stable-worldmodel[train,env]`，以及传递依赖里的 `imageio` / `Pillow` / `h5py`）。下面所有命令都用它。

```bash
cd ~/am/wm/le-wm           # A500 上是 ~/junjxu/wm/le-wm
source .venv/bin/activate
```

venv 如需重建：

```bash
cd ~/am/wm/le-wm
uv venv --python=3.10
source .venv/bin/activate
uv pip install swig                       # box2d-py 在编译时需要 swig
cat > /tmp/lewm-build-constraints.txt <<'EOF'
setuptools<=66.0.0
wheel<=0.38.4
pip<=23.0.1
EOF
uv pip install --build-constraints /tmp/lewm-build-constraints.txt 'stable-worldmodel[train,env]'
uv pip install -U 'datasets>=2.14,<4'     # 上游 pin 太老
# 系统是 CUDA 12.9 driver，把默认 cu130 wheel 换成 cu128：
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128'
```

---

## 数据集

数据放 `$STABLEWM_HOME`（默认 `~/.stable_worldmodel/`）。`stable_worldmodel.HDF5Dataset` 在该目录下找 `<name>.h5`。

| 文件 | 来源 | 大小 | 备注 |
|---|---|---|---|
| `pusht_expert_train.h5` | LeRobot pusht（HF: `quentinll/lewm-pusht`）| 44 GB | `le-wm` 的参考数据集。2.3M 帧，18685 集，含 action + proprio + state |
| `phyworld_{uniform_motion,parabola}.h5`, `phyworld_collision_eval.h5` | 下面的转换脚本 | ~100 MB | **Eval 集**（全 4 partition：ID + r/m-OOD + v-OOD + both-OOD）。用于 probing / rollout |
| `phyworld_{collision,uniform_motion,parabola}_id1k.h5` | 转换脚本 + `--limit 1000` on `*_30K.hdf5` | ~100–160 MB | **纯 ID 训练集**（1000 traj，32k 帧，100% ID）。用于 leak-free ID→OOD FT。基于 HF `magicr/phyworld` `id_ood_data/*_30K.hdf5` |

转换脚本：[`convert_to_lewm.py`](./phyworld/scripts/convert_to_lewm.py)（uniform/parabola，action=速度），[`convert_collision_to_lewm.py`](./phyworld/scripts/convert_collision_to_lewm.py)（collision，action=加速度）。**action 语义因域而异** — uniform/parabola 用速度，collision 用加速度；LeWM 的 `action_encoder` 维度也要匹配（2 vs 4）。

---

## phyworld → le-wm 桥接

### [`convert_to_lewm.py`](./phyworld/scripts/convert_to_lewm.py) 做的事

phyworld 的 hdf5 把每条轨迹存为 MP4 字节流（`video_streams/<group>/<idx>`）+ 2D 位置（`position_streams/...`）。它**没有 action 信号**——是被动物理数据集。

le-wm 需要的是单个 flat-stacked hdf5，带 `pixels` / `action` / `proprio` / `ep_len` / `ep_offset` / `episode_idx` / `step_idx`。

脚本流程：

1. 解码每条轨迹的 MP4 → `(T, H, W, 3)` uint8
2. resize 到 `--img-size`（默认 224，对应 le-wm 的 `img_size`）
3. 直接用 `position[t]` 作 `proprio[t]`
4. **合成** `action[t] = position[t+1] − position[t]`（速度）。原因：le-wm 的 column normalizer 会除以 `std`；全零的 `action` 会 NaN。速度对匀速运动也是物理上合理的"控制信号"
5. 写 ep 簿记（每条 `ep_len = 32` 等）

从 `phyworld/` 目录跑（让默认 `--src` 解析到 `data/uniform_motion_eval.hdf5`）：

```bash
cd ~/am/wm/phyworld
python scripts/convert_to_lewm.py
# → 写到 ~/.stable_worldmodel/phyworld_uniform_motion.h5（~100 MB，2-3 分钟）
```

常用 flags：

```bash
python scripts/convert_to_lewm.py --limit 4 --dst /tmp/phyworld_test.h5  # 快速检查
python scripts/convert_to_lewm.py --img-size 256 --name phyworld_um256   # 自定义输出
python scripts/convert_to_lewm.py --src path/to/other_phyworld.hdf5      # 不同来源
```

### le-wm 的 phyworld 配置

已建好在 [`le-wm/config/train/data/phyworld.yaml`](./le-wm/config/train/data/phyworld.yaml)：

```yaml
dataset:
  name: phyworld_uniform_motion          # → ~/.stable_worldmodel/phyworld_uniform_motion.h5
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 1                           # phyworld traj 只 32 帧；不跳
  keys_to_load: [pixels, action, proprio]
  keys_to_cache: [action, proprio]
```

vs `pusht.yaml` 的差别：`frameskip=1`（原本 5），因为轨迹短；没有 `state` key（phyworld 没有 env 真值状态）。

---

## 跑训练

### Smoke test — 约 10 秒验证整条流水线

```bash
cd ~/am/wm/le-wm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python train.py \
  data=phyworld \
  output_model_name=lewm_phyworld \
  wandb.enabled=False \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=1
```

成功的样子：

```
Trainer.fit stopped: `max_epochs=1` reached.
Epoch 0/0 ━━━━━━━ 2/2 0:00:03
fit/pred_loss = 0.24    fit/sigreg_loss = 40.0
```

如果要换回 pusht：把 `data=pusht`，删掉 `output_model_name`（或换个名字）。

### 完整训练

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
  data=phyworld \
  output_model_name=lewm_phyworld \
  wandb.enabled=False
# ckpt 落到 ~/.stable_worldmodel/lewm_phyworld_epoch_<N>_object.ckpt
```

要开 W&B：编辑 [`le-wm/config/train/lewm.yaml`](./le-wm/config/train/lewm.yaml) 设 `wandb.config.entity / project`，再去掉 `wandb.enabled=False`。

### ⚠️ 启动训练时必检的一行

```
[init_from_ckpt] loaded=216 unexpected=0
```

任何 `loaded < 200` 都说明 ViT 命名 mismatch，整批训练作废。[`le-wm/train.py`](./le-wm/train.py) 已加 `_remap_old_vit_keys()` + 加载守卫，正常情况下不会出问题——但每次重跑都要看 log 确认这行。

---

## Deep-supervision probe（PIWM 风格）— 已加入 `lewm.yaml`

Recipe 来自 *Improving World Models using Deep Supervision with Linear Probes*（arXiv:2504.03861），实现的是 PIWM 原则 1（arXiv:2412.12870）。LeWM FT 时额外加一个 linear probe loss，让 projector-space 的 emb 与物理状态对齐。默认关闭 → 不影响 baseline；做消融时打开。

配置块在 [`le-wm/config/train/lewm.yaml`](./le-wm/config/train/lewm.yaml)：

```yaml
loss:
  probe:
    enabled: false          # true = 加 probe 臂
    weight: 1.0
    target: proprio         # 单列或列表，如 [proprio, action]（pos+vel）
    frames: 1               # 1 = 单帧 probe；K>1 = 堆 K 帧（让速度可解码）
```

关键发现见 [`reports/5-26/piwm_deepsup_results.md`](./reports/5-26/piwm_deepsup_results.md)：**单帧** probe（`frames=1`）有助于位置 / vy / 长程 cos，但**伤害高速 OOD 的 vx**（单帧编不出瞬时速度）。**多帧**（`frames=4`，`target=[proprio,action]`）是解药——能找回 vx 同时取得最好的长程 rollout cosine。

例子：

```bash
cd ~/lewm_run && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u train.py \
  data=phyworld_parabola_id1k loss.probe.enabled=true \
  'loss.probe.target=[proprio,action]' loss.probe.frames=4 \
  output_model_name=lewm_parabola_piwm_mf4_id1k subdir=parabola_piwm_mf4_id1k \
  trainer.max_epochs=20 +init_from_ckpt=~/.stable_worldmodel/lewm_paper_pusht/weights.pt
```

Eval / AR-rollout：[`phyworld/scripts/rollout_eval_id1k.py`](./phyworld/scripts/rollout_eval_id1k.py)（`--ckpt` / `--tag` 切换臂；输出 latent cos vs horizon/partition + K=1 / K=4 解码的 pos/vel ρ）。

### ⚠️ 但 K=4 ρ / latent cos 不能单独信

详见诊断报告 [`reports/6-24/diagnostic_report.md`](./reports/6-24/diagnostic_report.md)。简单说：

- **K=4 ρ 是 probe loss 的对偶**：probe loss 越小，ρ 必然越大——这是数学必然，不是 world model 变好
- **latent cos** 同 ckpt 的 encoder 编码两端（real 和 pred），是循环逻辑：encoder 塌方时两端都被压扁到同一低维流形，cos 必然高
- **真正可信的主指标是 `validate/pred_loss_epoch`**（next-state prediction loss）。选 ckpt / 选 (w, f) / 报告论文 main table 都用它。其他指标只作为次要诊断

---

## Caveats / 已知问题

- **collision 上 auto-resume 易出锅**：le-wm 如果发现 `~/.stable_worldmodel/<output_model_name>_weights.ckpt` 存在会自动 resume。在数据集之间切 `data=` 会改 action_dim → state_dict shape mismatch。每个任务用不同的 `output_model_name`（上面的例子就是这么做的）
- **phyworld 不是真正的训练集**：36k 帧单球匀速运动对 WM 训练太少 + 太简单；它是 *evaluation* 集。要做真训练需用 `phyworld/id_ood_data/*.py` 生成更大集合（30k–3M 视频）再跑转换器
- **action 是合成的**：phyworld 上 model 实际学的接近"无条件 next-frame prediction"。**不要**把 loss 曲线解读为"le-wm 在 phyworld 上学到了物理"——参见 [`phyworld/README.md`](./phyworld/README.md) 中预期的评估协议
- **A500 init bug 已修但要 verify**：详见上面"启动训练时必检的一行"
