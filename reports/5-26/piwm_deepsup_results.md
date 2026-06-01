# 实验 A：PIWM-style Deep-Supervision Linear Probe

**日期**：2026-05-31
**测试域**：⚠️ **仅 parabola（抛物线，自由落体 + 重力）一个域**。collision / uniform_motion 尚未做（§8 待办）。所有数字都只针对 parabola。
**动机**：[rollout_results.md](rollout_results.md) 暴露两个问题——(1) AR rollout 的 latent cosine 随 horizon 衰减（长程漂移），(2) §4 从 rollout latent 解码 pos/vel 偏弱。本实验借鉴 PIWM 思路（arXiv:2412.12870 + deep-supervision arXiv:2504.03861），在 LeWM FT 时加一项 **linear probe 监督 loss**，看能否同时改善这两点。
**脚本**：训练 [le-wm/train.py](../../le-wm/train.py)（加 `loss.probe`）；评估 [rollout_eval_id1k.py](../../phyworld/scripts/rollout_eval_id1k.py)

---

## 1. 方法

在 LeWM FT 的 loss 里加一项 deep-supervision linear probe（默认关，`loss.probe.enabled` 开关，方便消融）：

```python
# train.py  lejepa_forward
probe_pred = self.model.probe_head(emb)              # Linear(192 → proprio_dim)
probe_loss = (probe_pred - proprio_normalized)**2    # 监督 position
loss = pred_loss + λ_sigreg·sigreg + λ_probe·probe_loss
```

- `probe_head` = **单层 Linear(192 → 2)**，对齐 deep-supervision 论文（不是 PIWM 的 MLP extractor）
- 监督目标 = **proprio（position，z-score 归一化）**，对齐 PIWM intrinsic 的 `state[:,:2]`
- 在 **projector 空间**做（predictor 的工作空间），所以 rollout 预测的 latent 也受益
- `λ_probe = 1.0`

**配置**（跟 baseline 完全一致，只多 probe 项）：
- data = `phyworld_parabola_id1k.h5`（PhyWorld 官方 ID-only 1000 trajs）
- init = pusht weights, 20 epoch, GPU 3
- baseline 臂 = `parabola_paperinit_id1k`（`loss.probe.enabled=false`）

**训练有效性验证**：`probe_loss` 从 **1.475 → 0.036**（降 40×），`pred_loss` 0.0046（无退化）。说明 emb 被成功逼成"线性可读 position"，且没破坏原始预测任务。

---

## 2. 结果：latent cosine vs rollout horizon（长程漂移）

aggregate over test trajs：

| horizon | baseline | +probe | Δ |
|---|---|---|---|
| h=1 | 0.978 | 0.984 | +0.006 |
| h=2 | 0.932 | 0.958 | +0.026 |
| h=4 | 0.864 | 0.917 | **+0.053** |
| h=8 | 0.814 | 0.877 | **+0.063** |
| h=16 | 0.535 | 0.633 | **+0.098** |
| h=28 | —† | —† | 数值发散尾，不可比 |

† 两臂 h=28 都有少数 traj 的 AR rollout 数值发散（nMSE 爆炸到 1e7~1e8），cosine 不稳，略去。

→ **中长程 horizon 漂移被显著压住**（h=8 +0.063, h=16 +0.098）。

---

## 3. 结果：per-partition latent cosine（ID→OOD，最强信号）

aggregate over all horizons，test trajs：

| partition | baseline | +probe | Δ |
|---|---|---|---|
| ID | 0.935 | 0.978 | +0.043 |
| r-OOD | 0.741 | 0.886 | **+0.145** 🔥 |
| v-OOD | 0.581 | 0.701 | **+0.120** 🔥 |
| both-OOD | 0.511 | 0.658 | **+0.147** 🔥 |

→ **OOD 上的提升比 ID 还大**（+0.12~0.15）。deep-supervision 让 rollout 在没见过的 OOD partition 上漂移慢得多——这是本实验最强的结果，说明把 position 钉进 emb 让 predictor 的外推更稳健。

---

## 4. 结果：从 rollout latent 解码 position（§4 旧弱项）

probe 应用到 **predicted（rollout）emb**，per-partition：

| partition | baseline pos_x | +probe pos_x | baseline pos_y | +probe pos_y |
|---|---|---|---|---|
| ID | 0.633 | **0.948** 🔥 | 0.667 | **0.978** 🔥 |
| r-OOD | 0.756 | 0.892 | 0.794 | 0.967 |
| v-OOD | 0.634 | 0.853 | 0.803 | 0.907 |
| both-OOD | 0.646 | 0.806 | 0.760 | 0.911 |

→ **position 解码大幅改善**（+0.15~0.32），ID position 从 ~0.63 直接到 ~0.96。因为 probe loss 监督的就是 position，emb 高度线性可读，rollout 后仍保持。这正面解决了 rollout_results §4 "解码弱" 的问题。

---

## 5. velocity：vy 被救起，vx 在高速 OOD 上反被 probe 牺牲

**K=1（单帧）解码 velocity 几乎失败**——单帧本就读不出速度（§6 早证 vel 需多帧差分）。

**修法：K=4——把 rollout 预测出的连续 4 个 latent 拼起来再解码**（`[ê_{k-3},…,ê_k]`，跟 §6 在真实 emb 上的 K=4 同款，只是作用在预测 latent 上）。澄清：rollout **能**做 K=4，predictor 吐的是整条预测序列，拼 4 帧即可；唯一真约束是必须留在 projector 空间（projector 不可逆）。

### 5.1 vy（垂直，重力驱动）—— deep-sup + K=4 都帮忙

从 rollout 预测 latent 解码 **vy ρ**（per-partition）：

| partition | baseline K=1 | baseline K=4 | +probe K=1 | +probe K=4 |
|---|---|---|---|---|
| ID | 0.734 | **0.870** | 0.983 | **0.987** |
| r-OOD | 0.873 | 0.834 | 0.978 | 0.979 |
| v-OOD | **0.413** | **0.873** 🔥 | 0.912 | 0.923 |
| both-OOD | 0.603 | **0.794** | 0.903 | 0.935 |

→ vy 上 K=4 和 +probe **都正向**：baseline v-OOD K=1→K=4 从 0.41→0.87；+probe 进一步抬到 0.92。原因：重力让 vy 跟 pos_y 强耦合，监督 position 顺带帮了 vy。

### 5.2 vx（水平，常量速度）—— +probe 在高速 OOD 上**反而更差**

从 rollout 预测 latent 解码 **vx ρ**（per-partition）：

| partition | baseline K=1 | +probe K=1 | baseline K=4 | +probe K=4 |
|---|---|---|---|---|
| ID | 0.328 | 0.287 | 0.304 | **0.585** ✅ |
| r-OOD | 0.407 | 0.462 | 0.612 | 0.662 ✅ |
| v-OOD | **0.584** | 0.413 ❌ | **0.649** | 0.450 ❌ |
| both-OOD | 0.527 | 0.603 | **0.696** | 0.515 ❌ |

→ **这是 +probe 的一个真实代价，不是噪声**：
- ID / r-OOD 上 +probe 帮了 vx（K=4: ID 0.30→0.59, r-OOD 0.61→0.66）
- **但高速 partition（v-OOD / both-OOD）上 +probe 把 vx 拉低 0.18~0.20**（K=4: v-OOD 0.649→0.450, both-OOD 0.696→0.515），K=1 上同样（v-OOD 0.584→0.413）

**机制解读**：probe **只监督 position，且只在 ID 数据（vx∈[1,4]）上**。把 emb 压成 "position 线性可读" 的重组对 vy 有利、对解耦的 vx 不利——尤其外推到没见过的高速 OOD，vx 这种细微速度区分被挤压。**监督 position ≠ 监督 velocity，对 vx 甚至是负迁移**。

### 5.3 结论

1. **K=4 对 velocity 普遍有用**——多帧差分恢复速度信号（baseline v-OOD vy 0.41→0.87）。验证 "vel 需要多帧差分" 在 rollout 上成立。
2. **deep-supervision 不是免费午餐**：监督 position 帮了 position + vy，**但在高速 OOD 上牺牲了 vx**。"全面改善" 是过头的说法——vx 高速 OOD 是反例。
3. **caveat**：K=4 的 velocity 改善集中在短/中 horizon；长 horizon（h≥16）4 帧都已漂移，K=4 差分跟着噪，仍出 nan/负 ρ。

**该做的修正**：把 `loss.probe.target` 扩成 pos+vel（训练时**直接监督速度**），预期能消除对 vx 的负迁移——这是 §8 的首要待办。

---

## 6. 小结

| 指标 | 效果 |
|---|---|
| 长程 cos（h=8 / h=16）| ✅ +0.063 / +0.098 |
| OOD cos（r/v/both-OOD）| ✅ **+0.12 ~ +0.15（最强）** |
| 解码 position | ✅ **+0.15 ~ +0.32，ID 到 0.96** |
| 解码 vy | ✅ +probe 帮忙（重力让 vy 耦合 pos_y）；K=4 进一步救（baseline v-OOD 0.41→0.87）|
| 解码 vx | ⚠️ **混合**：ID/r-OOD 帮，**高速 OOD（v/both）+probe 反降 0.18~0.20** |
| pred_loss | ✅ 无退化（0.0046）|

> **结论**：PIWM-style deep-supervision linear probe 在 parabola 上**改善了长程 rollout 余弦相似度、position 解码、以及 vy 解码**——命中 rollout_results 的主要目标。**但不是免费午餐**：probe 只监督 position，对解耦的 **vx 在高速 OOD partition 上是负迁移**（−0.18~0.20）。K=4（拼 4 个预测 latent）对 velocity 普遍有用，但救不了 vx 的这个退化。这验证了 PIWM 原则 1（latent 对齐物理量）有效，同时暴露其局限——**只对齐 position 会偏科**；正确做法是训练时同时监督 pos+vel（§8 首要待办）。如需进一步压长程漂移，可上原则 2（physics-structured dynamics 替换 ARPredictor）。

---

## 7. 复现 & 数据

| 项 | 路径 |
|---|---|
| 训练（+probe）| `cd ~/lewm_run && CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u train.py data=phyworld_parabola_id1k loss.probe.enabled=true loss.probe.weight=1.0 loss.probe.target=proprio output_model_name=lewm_parabola_piwm_probe_id1k subdir=parabola_piwm_probe_id1k trainer.max_epochs=20 +init_from_ckpt=.../lewm_paper_pusht/weights.pt` |
| +probe ckpt | `~/.stable_worldmodel/parabola_piwm_probe_id1k/lewm_parabola_piwm_probe_id1k_epoch_20_object.ckpt` |
| baseline ckpt | `~/.stable_worldmodel/parabola_paperinit_id1k/...epoch_20...` |
| 配置开关 | `le-wm/config/train/lewm.yaml` → `loss.probe.{enabled,weight,target}`（默认 enabled=false）|
| rollout log（+probe）| /tmp/rollout_parabola_piwmprobe.log |
| rollout log（baseline）| /tmp/rollout_parabola.log |
| K=4 对比 log（baseline vs +probe）| /tmp/rollout_k4_cmp.log |
| 训练 log | /tmp/lewm_parabola_piwm_probe_train.log |
| K=4 decode 实现 | [rollout_eval_id1k.py](../../phyworld/scripts/rollout_eval_id1k.py) 末尾 K=4 block（`--ckpt` / `--tag` 可切模型）|

注：路径前缀 `agent_memory` 已重命名为 `am`（`/home/qlib/am/wm/...`）。

## 8. 待办（未做）

- [ ] velocity 训练时监督版（`target` 扩 pos+vel），看能否连长 horizon velocity 也救起来
- [ ] 推广到 collision / uniform_motion 两域
- [ ] 正式消融：同一域跑 `enabled=false` vs `true`（当前 baseline 用的是更早的 `parabola_paperinit_id1k`，严格消融应在同 commit 下重跑 enabled=false）
- [ ] λ_probe sweep（0.1 / 1.0 / 10.0）
