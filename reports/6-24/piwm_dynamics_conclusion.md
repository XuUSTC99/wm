# 引入运动学方程为何没提升 —— PIWM 论文对照结论

**日期**：2026-07-05
**问题**：在 le-wm 里给位置 slot 引入二阶运动学方程，期望改善外推 / 长程 rollout。首版（structdyn，可学 accel）latent 指标为负面/null。本报告对照 PIWM([2412.12870](../../papers/2412.12870.pdf)) 找出根因与解法。

---

## 0. TL;DR（三域 20+ run 实证，最终结论）

**真正让效果提升的是 free-rollout 训练（去掉 teacher forcing），不是运动学方程。** 这个结论在 **uniform / collision / parabola 三个域一致成立**：

1. **free-rollout = 决定性杠杆，三域全部大幅提升**（both-OOD nMSE：uniform 0.31→0.13、collision 1.11→0.39、parabola 0.79→0.31；长程 cos 普遍 +0.3 以上）。**直接印证 PIWM §4.1：teacher forcing 掩盖误差累积、free-rollout 才治长程漂移。**
2. **运动学方程在 free-rollout 之上不再带来提升，甚至变差**：uniform 上~中性（pixel/latent 与纯 FR 持平）；**collision 明显变差**（both-OOD 0.39→0.56，光滑 accel MLP 学不动碰撞冲量）；parabola 略差。
3. **pixel 尺（位置权重高）也是同结论**：纯 free-rollout(baseline_fr) 长程 pixel 最好（h28=22.1dB），加编码+运动学并不超过它。

**对最初目标的回答**：加运动学方程**没能**带来提升；固定编码+运动方程"两个一起"是 PIWM 的方向、但在当前 JEPA 形态下不 work。**要提升就上 free-rollout。**（详细归因见 §1-3 PIWM 对照。）

---

## 1. PIWM 到底靠什么拿到提升（原文证据）

### 1.1 free-rollout 训练，明确反对 teacher forcing（§4.1）

> "all models including baselines are trained using **free-rollout prediction without teacher forcing**. ... teacher forcing can **mask compounding errors** during training, leading to **unstable rollouts** under deployment conditions."

- PIWM 训练时就把预测喂回去、跑多步，让动力学学会"不累积误差"。
- **le-wm 现状**：`wm.num_preds=1`，`lejepa_forward` 里 `pred_emb=predict(ctx_emb)`、`tgt_emb=emb[:,1:]`，每步都用 ground-truth 上下文 → **纯 teacher-forced 单步**。
- 后果：黑盒视觉 slot 从没在训练中见过自身误差累积，eval 自回归必然漂。**train/eval 失配**（训单步、评自回归）。

### 1.2 extrinsic 解耦：物理态是独立主 latent（§4.2）

> "decoupling visual perception from physical state inference (the extrinsic approach) is a **critical design choice** for achieving robust, long-term prediction."
> DVBF 失败因为 "no physical inductive bias, causing the learned states to **drift and quickly accumulate errors**."

- PIWM 的 `z*` 是低维物理态，dynamics 演化它、decoder 从它重建。物理态是承重主体。
- **le-wm 现状**：位置 = 192 维 latent 里的前 2 维。pred_loss 对 192 维平均 → 那 2 维只占 ~1%，修得再好也被 190 维黑盒淹没；decoder 也没被强制只依赖 slot。→ **物理通道被稀释、非承重**。

### 1.3 固定形式动力学、只学参数（§3.2）

> "our prediction is based on **known dynamics equations** φ(z*,a,θ), where the form of φ is **fixed** and only its parameters θ are learnable."

- PIWM 的 φ 是写死的运动学，只学质量/摩擦。强物理先验、不会过拟合。
- **le-wm 首版**：accel 是自由 MLP。诊断发现它没停在 0（uniform 真实 a=0），学出 ~0.5×|v| 的乱修正 → **过拟合 ID 残差，毁掉 v-OOD/长程泛化**。

### 1.4 其他（次要）

- 离散量化 latent 作强正则（extrinsic-discrete 最优）；分阶段冻结训练；弱分布监督。

---

## 2. 首版 structdyn 负面结果的归因

| 观察（latent 指标 vs baseline/structpos） | 归因 |
|---|---|
| v-OOD nMSE 0.150→0.245 退化 | accel 自由 MLP 过拟合（§1.3） |
| 长程 h=16/28 cos 掉到 baseline 下 | ①单步 teacher-forced（§1.1）+ ②accel 累积误差 |
| both-OOD 仅微好、没回 baseline | ②物理通道稀释（§1.2） |
| 聚合指标整体没动 | 2/192 稀释 + 测量尺不对 |

---

## 3. 结论：怎么解决（按 PIWM 杠杆排序）

| 优先级 | 措施 | 对应差距 | 状态 |
|---|---|---|---|
| **① 最高** | **free-rollout 多步训练**：`num_preds>1`，unroll K 步把预测喂回、每步监督 | §1.1 | 待实现，杠杆最大 |
| ② 高 | **约束动力学**：`learnable_accel=false`(纯匀速) 或 `accel_reg` 正则；碰撞域才放开 | §1.3 | ✅ 正在跑 |
| ③ 中 | **物理通道承重**：extrinsic 解耦 / 预测直接在低维物理态上做 / 强制 decode 依赖 slot | §1.2 | 待定，较大改动 |
| ④ 测量 | **pixel rollout by horizon**（位置权重高），别只看 latent 聚合 | 稀释 | 待补 decoder |

**"两个一起"的定位**：固定编码(structpos) + 运动方程(dynamics) 是 PIWM 的 (i)+(ii)，**必须一起**；但要真见效，必须再叠加 ①free-rollout 和（②约束动力学 / ③承重通道）。单靠"两个一起"在当前形态不够。

---

## 4. 8 卡消融实测结果（2026-07-05，uniform_motion，latent nMSE↓ / h28 cos↑）

| run | 配置 | ID | r/m-OOD | v-OOD | **both-OOD** | **h28 cos** |
|---|---|---|---|---|---|---|
| baseline* | 无 struct 无 dyn, TF | 0.065 | 0.385 | 0.150 | 0.295 | ~0.84 |
| structpos* | struct, TF | 0.071 | 0.360 | 0.175 | 0.407 | — |
| structdyn | struct+可学accel, TF | 0.069 | 0.361 | 0.245⚠️ | 0.398 | 0.825 |
| structcv | struct+纯匀速, TF | 0.056 | 0.461 | 0.154 | 0.339 | 0.802 |
| structdyn_areg1 | +accel_reg=1 | 0.058 | 0.453 | 0.150 | 0.321 | 0.819 |
| structdyn_areg10 | +accel_reg=10 | 0.058 | 0.501 | 0.184 | 0.383 | 0.788 |
| structdyn_frzenc | +freeze_encoder | 0.116 | 0.457 | 0.434 | 0.526 | 0.789 |
| dynonly | dyn 无 struct | 0.160 | 0.387 | 0.232 | 0.376 | 0.797 |
| structcv_frzenc | 匀速+冻结 | 0.232 | 0.809 | 0.492 | 0.621 | 0.627 |
| **structpos_fr** | **struct+free-rollout** | **0.019** | **0.234** | **0.029** | 0.183 | **0.942** |
| **structcv_fr** | **struct+匀速+free-rollout** | 0.016 | 0.350 | 0.034 | **0.143** | 0.907 |

\*baseline/structpos 来自 [structured_loss_report.md]（同评测脚本）。

**读表**：
- **free-rollout（最后两行）整体碾压**：长程 cos 0.94/0.91 vs 其余全在 0.79–0.83；both-OOD/v-OOD 全面超 baseline。
- 纯匀速/正则（structcv/areg1）把可学 accel 的 v-OOD 退化(0.245)修回 ~0.15 ✓，但不超 baseline。
- **冻结编码器崩了**（frzenc/cvfrz）：pusht 初始化没适配 phyworld 域，别冻。
- **dynonly ID 崩到 0.16**：印证固定编码是 slot 有意义的前提。

## 5. uniform latent 2×2（隔离 free-rollout vs 固定编码）

同代码同评测，nMSE↓ / h28 cos↑：

| | teacher-forced | **free-rollout** |
|---|---|---|
| baseline(无编码) | both-OOD 0.308, h28 0.854 | both-OOD **0.131**, h28 **0.969** |
| structpos(有编码) | both-OOD 0.407, h28 — | both-OOD 0.183, h28 0.942 |

- 横看：**free-rollout 把 both-OOD 砍半、长程 cos 拉到 0.94–0.97**（决定性）。
- 纵看：**加固定编码在 FR 之上反而略差**（0.131→0.183）。纯 FR(baseline_fr) 就是最好。

## 6. uniform pixel by-horizon（位置权重高的"对的尺"，PRED PSNR dB↑）

| | h1 | h8 | h16 | h28 | both-OOD |
|---|---|---|---|---|---|
| structpos (TF) | 27.81 | 20.73 | 20.39 | 19.29 | 19.23 |
| **baseline_fr (纯FR)** | 25.55 | 21.05 | 21.86 | **22.09** | 20.41 |
| structpos_fr (FR+编码) | 26.25 | 20.86 | 20.74 | 20.24 | 19.17 |
| structcv_fr (FR+编码+运动学) | 26.04 | 21.55 | 22.17 | 21.06 | 20.49 |

- **纯 free-rollout 长程 pixel 最好**（h28 22.1）。
- 加编码(structpos_fr)在长程 pixel 反而掉（20.24）；运动学(structcv_fr)只是把编码造成的损失**修回**到 ~baseline_fr 水平，**并未超过纯 FR**。
- 先前"运动学 +1.4dB"是相对 structpos_fr 而言，对齐真正基线 baseline_fr 后**运动学净增益≈0**。

## 7. 跨域复现（collision / parabola，latent nMSE↓，both-OOD | h28 cos）

| 域 | baseline_tf | baseline_fr | structdyn_fr(带运动学) |
|---|---|---|---|
| uniform | 0.308 \| 0.85 | **0.131 \| 0.97** | 0.143 \| 0.91 |
| collision | 1.114 \| 0.25 | **0.393 \| 0.58** | 0.560 \| 0.48 ⚠️ |
| parabola | 0.786 \| 0.58 | **0.313 \| 0.95** | 0.354 \| 0.96 |

- **free-rollout 三域全部大幅提升**（both-OOD 砍半以上，collision 尤甚 1.11→0.39）。
- **运动学三域均无正贡献**：uniform~平、**collision 明显变差**（0.39→0.56，冲量非光滑、smooth accel MLP 学不动）、parabola 略差。
- (parabola h28 nMSE 有个别轨迹数值爆掉→用 cos 看更可靠，cos 两者都 ~0.95。)

## 8. 最终判断与下一步

**判断**：运动学方程（当前 2/192 slot + smooth accel 形态）不是提升来源，free-rollout 才是。要按 PIWM 真正吃到"物理结构"的红利，需要的是 **extrinsic 解耦（物理态做承重主 latent）+ 固定形式动力学（非自由 MLP）**，而非在黑盒 JEPA 上挂 slot。

**建议**：
1. 把 **free-rollout 设为默认训练模式**（`wm.free_rollout=true, num_preds≥8`）——这是已验证的净提升，三域通用。
2. 运动学方向若要继续，改走 extrinsic：低维物理态 + 物理方程 predictor，而不是当前的 slot 挂载。
3. collision 的 smooth accel 学不动冲量 → 若坚持 intrinsic，需要非光滑/接触感知的动力学项。

产物：`/data1/likun-share/junjxu/runs/structdyn_eval/{train_,rollout_,decoder_,pixel_,pxroll_}*`；脚本 [run_structdyn.sh](run_structdyn.sh)（参数化 GPU/NAME/DATA/DOM/SW/DYN/EXTRA）、[run_pixel.sh](run_pixel.sh)。
