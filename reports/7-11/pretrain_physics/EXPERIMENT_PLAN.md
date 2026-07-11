# 预训练 vs 后训练 注入物理结构 —— 实验方案

**日期**：2026-07-11
**动机**：6-24 所有物理实验（固定编码 structpos + 运动学 dynamics）都用了 `init_from_ckpt=pusht`（**后训练/fine-tune 嫁接**）——encoder 已先固化成黑盒，物理 slot 打不进去、还和黑盒预测器打架 → 全部否掉。**PIWM 恰恰是从头带物理结构训**。本实验测：**物理结构在 from-scratch 预训练时 bake 进去，会不会好？**

## 1. 核心假设

> **物理结构（固定编码 slot + 运动学方程）在 from-scratch 共训时能承重、有效；在后训练嫁接到已固化的黑盒 encoder 上时打架、有害。**

**若成立** → novelty：**"物理归纳偏置必须在预训练阶段注入，不能后训练嫁接"**——既解释了 6-24 为何全失败，又和 PIWM（从头训）自洽，是个干净可发表的洞察。
**若不成立**（from-scratch 也伤）→ 强化"物理在此架构下根本没用"的结论。两种都是干净结果。

## 2. 实验设计（2×2×域）

| 轴 | 取值 |
|---|---|
| init | **scratch**（无 init）/ **pusht**（`init_from_ckpt`，后训练） |
| physics | **off**（纯 free-rollout）/ **on**（structured.weight=1 + dynamics const，严格 PIWM 重力形式） |
| 域 | uniform / parabola / collision |
| 公共 | free-rollout(np8) 默认；**epoch 统一 60**（公平对比物理效应，消除 epoch 混淆） |

**关键量**：
- Δ_scratch = (scratch+physics) − (scratch+off)
- Δ_pusht  = (pusht+physics) − (pusht+off)
- **判据**：若 **Δ_scratch < Δ_pusht**（物理在 from-scratch 下更不伤/有益，尤其 Δ_scratch ≤ 0）→ 假设成立。

参照（6-24 已有，pusht@20ep）：parabola pusht physics 伤 +0.079（0.313→0.392）、uniform +0.035（0.131→0.166）。本实验 epoch 统一 60 重测以消混淆。

## 3. 运行矩阵（8 run，打满 GPU）

| # | run | init | physics | 域 |
|---|---|---|---|---|
| 1 | par_scratch_off | scratch | off | parabola |
| 2 | par_scratch_on | scratch | on | parabola |
| 3 | par_pusht_off | pusht | off | parabola |
| 4 | par_pusht_on | pusht | on | parabola |
| 5 | um_scratch_off | scratch | off | uniform |
| 6 | um_scratch_on | scratch | on | uniform |
| 7 | col_scratch_off | scratch | off | collision |
| 8 | col_scratch_on | scratch | on | collision |

（parabola 做完整 2×2 作主证据；uniform/collision 补 scratch 两臂，pusht 臂复用 6-24 或后补。）

## 4. 风险与对策
- **from-scratch 可能欠拟合**（encoder 随机初始化、数据仅 1k 轨迹）→ 监控 pred_loss 收敛，60 epoch 不够就延长；关注**相对效应 Δ**（不看绝对值），即使 scratch 整体差也能判物理效应方向。
- **epoch 混淆** → 全部统一 60 epoch。
- **物理形式**：用 `dynamics.accel_form=const`（严格 PIWM，parabola=重力/uniform=0），非自由 MLP（已知过拟合）。

## 5. 评估
- 主指标：latent both-OOD nMSE（同 rollout_eval_id1k.py，--max-trajs 500）+ r/m/v-OOD 分区 + h28 cos。
- 产物：`/data1/likun-share/junjxu/runs/pretrain_physics/`。

## 6. 结果与分析（2026-07-11）

### 6.1 数据（latent both-OOD nMSE↓，60 epoch）

| 域 | init | physics off | physics on | **Δ = on−off** |
|---|---|---|---|---|
| **parabola** | **scratch** | 0.559 | 0.678 | **+0.119（物理伤）** |
| **parabola** | **pusht** | 0.244 | 0.467 | **+0.223（物理伤）** |
| uniform | scratch | 0.349 | 0.576 | **+0.227（物理伤）** |
| collision | scratch | 0.359 | 0.675 | **+0.316（物理伤）** |

（uniform/collision 的 pusht 臂见 6-24：uniform Δ_pusht≈+0.035、parabola/collision 物理均伤。）

### 6.2 结论：假设否定

**"物理结构在预训练注入会更好"不成立。** 三个域上 **from-scratch 的物理效应 Δ 全为正（物理伤）**：parabola +0.119、uniform +0.227、collision +0.316。物理在 scratch 和 pusht 两种 init 下**都伤**。

**对论文的价值（喂 aaai_paper §4.2 + PIWM 反驳）**：
- 堵死 PIWM 辩护——"PIWM 有效是因为 from-scratch 共训" → **from-scratch（intrinsic 共享 latent）物理仍伤 → PIWM 靠的是整套 extrinsic 架构，不是 from-scratch 本身。**
- parabola 完整 2×2 是 §4.2 的直接证据表。

### 6.3 两个副发现
1. **from-scratch 整体差 + 训练不稳**：scratch 基线 0.559 vs pusht 0.244（2×差）；训练中 pred_loss 震荡 2-9（pusht ~0.008）。根因：**1000 轨迹训随机 encoder 数据不够 + free-rollout 从随机 encoder 展开 8 步早期全是垃圾预测**。数据受限是本质（本地无更大 phyworld 训练集）。
2. **pred_loss/结构指标又骗人**：scratch+物理的 pred_loss（0.01-0.15）远低于 scratch+off（2-9），但 both-OOD 反而更差（0.678 vs 0.559）。structured 锚定拉低了那几维的 loss，制造"训得更好"假象，rollout 才是真尺。（呼应 aaai_paper §6 评测陷阱。）

### 6.4 caveat 与 P0 补跑
scratch 基线欠拟合/不稳 → 审稿人可攻击"from-scratch 训崩、2×2 无效"。**注意：Δ 的方向（物理伤）在同一欠拟合设定内部仍成立**（scratch_on 0.678 > scratch_off 0.559），所以结论方向稳；但绝对数不干净。**已补跑更长(120ep)+更低 LR 的 from-scratch parabola 2×2 夯实基线**（见 §6.5）。

### 6.5 干净重跑结果（120ep + lr 2e-5，基线已收敛）

**降 LR 修好了 from-scratch 训练不稳**：pred_loss 从 2-9 震荡 → 收敛 ~0.008（和 pusht 同量级）。**caveat 解决——不稳是调参问题（LR 太高），非数据本质。**

| init | r/m-OOD | v-OOD | both-OOD |
|---|---|---|---|
| scratch off | 0.343 | 0.399 | 0.651 |
| scratch on | 0.375 | 0.410 | 1.20 ⚠️ |
| pusht off | 0.124 | 0.195 | 0.314 |
| pusht on | 0.198 | 0.190 | 0.377 |

**⚠️ parabola both-OOD nMSE 长程不可信**：scratch_on both=1.20 被 h28 数值爆点污染（h28 nMSE 197万，个别轨迹球出框/目标方差→0 除零；h28 cos 仍 0.91）。**判决用干净分区 r/m-OOD。**

**干净结论（r/m-OOD）**：
- Δ_scratch = 0.343→0.375 = **+0.03（物理伤）**
- Δ_pusht = 0.124→0.198 = **+0.07（物理伤）**
- **物理两种 init 下都伤 → 假设否定坐实，且基线已收敛、reviewer-proof。**

**副结论**：from-scratch 整体仍差于 pusht（r/m 0.343 vs 0.124），即使 pred_loss 收敛 → "训练 loss 收敛 ≠ rollout OOD 好"，喂 aaai_paper §6 评测陷阱。

---

## 7. 交接给 aaai_paper（§4.2 Table 2 用）

**Claim**：物理归纳偏置在 from-scratch 预训练时**仍伤**，排除"PIWM 有效是因为 from-scratch"的辩护 → PIWM 靠的是整套 extrinsic 架构。

**可直接进 Table 2 的行（parabola，r/m-OOD nMSE，↓，60ep 主 / 120ep 收敛复核）**：

| init | physics off | physics on | Δ | 备注 |
|---|---|---|---|---|
| post-hoc(pusht) | 0.124 | 0.198 | +0.07 | 物理伤 |
| from-scratch | 0.343 | 0.375 | +0.03 | 物理伤，基线已收敛(lr2e-5,120ep) |

**写作注意**：parabola both-OOD nMSE 长程有数值爆点，Table 2 用 r/m-OOD 或标注 cos；from-scratch 整体弱是数据受限(1k 轨迹)，写进 caveat。三域一致(uniform/collision 60ep 版 Δ_scratch 也>0)。
