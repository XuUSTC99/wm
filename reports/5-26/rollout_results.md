# AR Rollout 实验结果（ARPredictor forward dynamics）

**日期**：2026-05-27
**脚本**：[rollout_eval_id1k.py](../../phyworld/scripts/rollout_eval_id1k.py)
**模型**：3 个 LeWM ID-only FT ckpt（`{collision,uniform,parabola}_paperinit_id1k`, 20ep, 仅 1000 ID trajs 训练）
**前置**：这是 [arpredictor_rollout_proposal.md](arpredictor_rollout_proposal.md) 提案的实现。区别于 §6 的 state-encoding probe——**这次真正调用了 ARPredictor 做自回归预测**。

---

## 1. 这次测的是什么（跟 §6 的区别）

| | §6 state-encoding probe | 本实验 AR rollout |
|---|---|---|
| 用 predictor? | ❌ 只用 encoder | ✅ encoder + ARPredictor |
| 流程 | `encoder(frame_t)` → probe 读当前 pos/vel | 编码前 3 帧 → predictor 自回归预测后续 28 帧 emb |
| emb 空间 | raw cls-token | **projector(encoder(x))**（predictor 实际工作空间）|
| action | 不需要 | **必须用 FT 时同款归一化**（ID-train mean/std）|

**关键实现点**（回应 "action 要不要跟 FT 一致"）：
- action 用 ID-only 训练集的 mean/std 做 `(x−mean)/std`，跟 [train.py](../../le-wm/train.py) 的 `get_column_normalizer` 完全一致
- OOD 的 action 归一化后本就超出 ±3，这是 OOD 的应有表现，没用 eval stats 偷拉回来
- predictor 在 projector 空间工作（projector 是带 BN 的 MLP，非 Identity），所以 probe 在 projector 空间重训

协议：history_size=3，每条 traj 编码前 3 帧真实 emb 当 context，之后**把预测 emb 喂回去**自回归 rollout 到第 32 帧。500 trajs/域，80% 训 probe / 20% 测。

**⚠️ action 定义三个域不一样**（已核对 train/eval/model 三方维度 + 定义一致）：

| 域 | action 定义 | 维度 | 含义 |
|---|---|---|---|
| collision | `(v[t+1]−v[t])/dt` **加速度** (force-as-action) | 4D (ax1,ay1,ax2,ay2) | predictor 必须积分 accel→vel→pos，**最干净的 forward dynamics 测试** |
| uniform_motion | `pos[t+1]−pos[t]` **速度** | 2D (vx,vy; vy≡0) | action 给了位移，部分泄漏运动 |
| parabola | `pos[t+1]−pos[t]` **速度** | 2D (vx,vy) | 同上，重力效应已在 vy 序列里 |

归一化各用各自 ID-train 的 mean/std（uniform/parabola 的 eval std ≈ 2.4× train std，因为含 v-OOD 高速球——归一化后 OOD action 偏大是应有表现）。**collision 用加速度这点很重要**：它是唯一要求 predictor 真正积分动力学的域，而它恰恰 rollout 漂移最快（§2），加强了 "学到单步、不会长程外推" 的结论。

---

## 2. 核心结果：latent fidelity vs rollout horizon

预测 emb 跟真实 emb 的 cosine similarity（test trajs, aggregate）：

| horizon | collision | uniform_motion | parabola |
|---|---|---|---|
| **h=1**（1-step）| **+0.99** | **+0.99** | **+0.98** |
| h=2 | +0.97 | +0.98 | +0.93 |
| h=4 | +0.86 | +0.96 | +0.86 |
| h=8 | +0.63 | +0.91 | +0.81 |
| h=16 | +0.34 | +0.84 | +0.53 |
| h=28 | +0.24 | +0.84 | +0.57 † |

† parabola h=28 的 nMSE 爆炸到 ~4e5（cosine 还有 0.57）——说明**少数 traj 的 AR rollout 数值发散**（latent norm 爆掉），是 AR 不稳定的典型表现。

**三个核心发现**：

1. **1-step forward model 非常准**（所有域 cos 0.98-0.99，含 OOD）。**ARPredictor 确实学到了局部动力学**——给定当前状态 + action，能准确预测下一帧 latent。

2. **多步 AR rollout 会复合误差、漂移**。漂移速度取决于动力学复杂度：
   - **uniform_motion 漂移最慢**（h=28 仍 cos 0.84）——匀速直线最好积分
   - **collision 漂移最快**（h=8 就掉到 0.63，h=28 仅 0.24）——碰撞事件本质难预测/混沌
   - **parabola 居中**，但有数值发散尾巴

3. **这是 num_preds=1 训练的必然结果**：LeWM FT 目标函数只优化 1-step 预测（[train.py:36-40](../../le-wm/train.py#L36-L40)），从没训过多步 rollout，所以多步漂移是预期的，不是 bug。

---

## 3. ID→OOD：rollout 在 OOD 上漂移更快

per-partition latent cosine（aggregate over all horizons，所以混了 drift 和 partition）：

| Partition | collision | uniform_motion | parabola |
|---|---|---|---|
| ID | +0.51 | +0.96 | +0.93 |
| r/m-OOD | +0.50 | +0.76 | +0.74 |
| v-OOD | +0.36 | +0.92 | +0.58 |
| both-OOD | +0.47 | +0.86 | +0.51 |

**观察**：
- **v-OOD / both-OOD（高速球）rollout 漂移明显更快**（parabola v-OOD 0.58 vs ID 0.93）。原因：速度越高 → 每步位移越大 → 积分越难，且 encoder/predictor 只见过 ID 速度范围
- uniform_motion 上 OOD 退化最小（v-OOD 仍 0.92）——匀速动力学最简单，外推到高速也还行
- **这是 ID→OOD generalization 在 rollout 上的体现**，比 §6 单帧 probe 上的退化更明显（单帧 probe 只读当前状态，rollout 要外推动力学）

---

## 4. 从 rollout latent 解码 pos/vel

把在真实 emb 上训的 Ridge probe 应用到 rollout 预测的 latent 上，per-partition（test）：

**Position ρ（rollout 后位置漂移慢，保持较好）**：

| | parabola REAL pos_y ρ | parabola PRED pos_y ρ |
|---|---|---|
| ID | +0.670 | +0.667 |
| r-OOD | +0.926 | +0.794 |
| v-OOD | +0.830 | +0.803 |
| both-OOD | +0.891 | +0.760 |

→ rollout 后的 latent 解出的**位置仍跟真值高度相关**（pos 漂移慢），OOD 上掉一些（0.89→0.76）。

**Velocity ρ：单帧解码本就弱**（§6 已知 vx 需要 K=4）。从单帧 rollout latent 解 vel 的 per-horizon 数值很不稳（h=28 出现 nan / 负 ρ），因为 (a) K=1 probe 对 vel 本就弱，(b) 长 horizon latent 已漂移。这个 metric 信息量低，不作为主要结论。

---

## 5. 结论

> **回到最初的问题 "能不能预测抛物线运动"**：
>
> - **1-step 能**：ARPredictor 在所有 3 个域上 1-step 预测 cos 0.98-0.99，确实学到了局部 forward dynamics（包括 OOD 初速/球大小）。
> - **多步外推会漂移**：AR rollout 复合误差，8-16 步后 latent 显著偏离。漂移速度 = f(动力学复杂度)：匀速最慢、碰撞最快、抛物线居中（且有数值发散尾）。
> - **OOD 上漂移更快**：高速 partition（v-OOD/both-OOD）rollout 误差累积明显快于 ID——说明 predictor 的动力学外推能力在 OOD 上确实下降。
> - **根因**：FT 只用 num_preds=1 的 1-step 目标，从没优化过多步 rollout。要真的"预测整条抛物线"得用 multi-step rollout loss 重训，或像 LeWM 论文那样接 planning（CEM 搜 action 到 goal）。
>
> **一句话**：LeWM ID-only FT 学到了**准确的单步动力学**，但**不是一个能长程外推的世界模型**——多步预测会漂移，OOD 上更甚。这跟 §6 "encoder 能编码当前 vy ρ=0.98" 是两件事：能编码当前状态 ≠ 能预测未来轨迹。

---

## 6. 局限 & 下一步

- **velocity 解码用了 K=1 probe**（弱）。可改成从连续 4 个 rollout latent 堆 K=4 再解 vel，会更公允。
- **没跑 frozen baseline 的 rollout**：frozen pusht predictor 没见过 phyworld 动力学，rollout 应该更差，可作对照（但意义有限）。
- **没跑真·planning**（给 start+goal 搜 action）：phyworld 是被动数据集，`swm.World` 里没有交互式 simulator，要跑得先搭 phyworld env，工程量大。
- **数值发散**（parabola h=28 nMSE 爆炸）：可加 latent norm clipping 或只报 h≤16。
- 数据 log：`/tmp/rollout_{collision,uniform,parabola}.log`

> **后续（已做）**：针对本报告暴露的"长程 cos 衰减 + 解码弱"两个问题，做了 PIWM-style deep-supervision 实验 → 见 [piwm_deepsup_results.md](piwm_deepsup_results.md)。结论：加 linear probe 监督 position 后，**长程 cos +0.06~0.10、OOD cos +0.12~0.15、position 解码 +0.15~0.32（ID 到 0.96）**；velocity 因未监督无改善。
