# 怎么让运动学方程真正帮到 OOD/长程 —— 探究

**日期**：2026-07-05
**背景**：三域实证已确认 free-rollout 是主升力、运动学在其上净贡献≈0（[piwm_dynamics_conclusion.md](piwm_dynamics_conclusion.md)）。本报告探究：**运动学到底有没有可用的价值 niche，怎么把它逼出来。**

---

## 1. 关键发现：运动学在"物理 OOD"上确实守住了位置

绕开 192 维聚合稀释，只看**解码位置 ρ**（位置通道本身），uniform：

| | v-OOD (x / y) | r/m-OOD (x) |
|---|---|---|
| baseline_fr（纯 FR） | 0.917 / 0.819 | 0.916 |
| structcv_fr（+运动学） | **0.975 / 0.973** ✅ | **0.288** ⚠️ |

**解读**：
- **v-OOD（速度分布外）运动学把位置守得最好** —— 匀速外推对任意速度精确，这正是它该赢的地方。y 方向 0.819→0.973 是大幅提升。
- **r/m-OOD（半径/质量变→外观变）位置 x 崩了** —— 外观一变，位置 slot 的编码被带偏，匀速外推把偏差放大。

**结论**：问题不在动力学，在 **① 位置编码对外观 OOD 不鲁棒；② 物理 slot 在 pred_loss 里只占 2/192、非承重**，好处被黑盒通道淹没。

---

## 2. 假设与探究设计

**假设**：把物理 slot 在 pred_loss 里**加重（pos_weight）→ 逼编码器/预测器把位置稳稳编进 slot、让它承重且线性于真实位置**，则匀速外推对得上、运动学能净超过纯 FR。

**代码**：新增 `loss.pos_weight`（[train.py](../../le-wm/train.py) pred_loss 对结构化 slot 维度加权）。

**sweep（8 卡并行，全 free-rollout 默认，uniform）**：

| arm | 运动学 | pos_weight |
|---|---|---|
| structpos_fr | 无（承重对照） | 10 / 30 / 100 |
| structcv_fr | 匀速 | 3 / 10 / 30 / 100 |
| structdyn_areg_fr | 可学+正则(accel_reg=1) | 10（为非匀速域铺路） |

对照：baseline_fr、structpos_fr(pw1)、structcv_fr(pw1) 已有。

**判据**：某配置的 both-OOD / 长程 latent nMSE 或 decoded-pos ρ **净超过 baseline_fr**（0.131 / pos ρ 0.92）。
- 若 pos_weight 让运动学净超 → 找到"怎么让运动学有用"：承重 + 匀速外推。
- 若仍不超 → 强证据表明 intrinsic-slot 形态下运动学无用，真正路径是 PIWM 的 **extrinsic**（低维物理态做主 latent + 固定形式动力学）。

---

## 3. 结果（uniform，latent nMSE↓ / 解码位置 ρ↑）

| run | ID | r/m | v | **both** | h28cos | posρ v-OOD(x/y) | posρ r/m(x) |
|---|---|---|---|---|---|---|---|
| **baseline_fr（纯FR，参照）** | 0.020 | 0.173 | 0.030 | **0.131** | 0.969 | 0.917/0.819 | 0.916 |
| structpos_fr (pw1) | 0.019 | 0.234 | 0.029 | 0.183 | — | 0.967/0.828 | 0.918 |
| structpos_fr **pw30** | 0.014 | 0.183 | 0.023 | **0.114** ✅ | 0.966 | 0.962/0.837 | 0.956 |
| structpos_fr pw100 | 0.022 | 0.237 | 0.032 | 0.136 | — | 0.961/0.836 | 0.958 |
| structcv_fr (pw1,有运动学) | 0.016 | 0.350 | 0.034 | 0.143 | — | 0.975/0.973 | 0.288⚠️ |
| structcv_fr pw10 | 0.018 | 0.311 | 0.030 | 0.175 | — | 0.988/0.983 | 0.578 |
| structcv_fr pw30 | 0.014 | 0.321 | 0.029 | 0.146 | 0.909 | 0.990/0.989 | 0.338 |
| structcv_fr **pw100** | 0.014 | 0.265 | 0.026 | **0.109** ✅ | 0.937 | **0.991/0.987** | 0.426 |
| structdyn_areg_fr pw10 | 0.017 | 0.221 | 0.031 | 0.149 | 0.928 | 0.965/0.938 | **0.755** |

## 4. 结论：找到让运动学净超纯 FR 的配方，但主升力是"承重编码"不是运动学

1. **承重（pos_weight≈30）本身就是净提升**：`structpos_fr_pw30`（承重、无运动学）ID/v/both-OOD 全面小胜 baseline_fr（both 0.131→0.114），长程持平，且**不带 r/m 崩**。**这是最干净的一档。**
2. **承重 + 运动学 = v-OOD 位置最强 + both-OOD 最低**：`structcv_fr_pw100` both-OOD **0.109（全场最低）**、v-OOD 位置 ρ **0.991/0.987（全场最高）**。运动学的价值在承重后被激活了——**匀速外推在速度 OOD 上确实精确**。
3. **但运动学仍有两个代价**：① **r/m-OOD（外观变）**始终是软肋（承重只把 posρ 从 0.29 救到 0.43，仍远低于无运动学的 0.96）——外观 shift 破坏 slot 编码，动力学放大它；② 192 维**聚合长程 cos 略降**（黑盒通道被扰动）。
4. **可学+正则 accel（areg）在 r/m 上更鲁棒**（posρ 0.755 vs 匀速 0.578）——因为能自适应，为非匀速域（collision/parabola）铺路。

**一句话**：让 OOD 变好的主力是"**把物理状态编码做成承重的**（pos_weight）"，不是运动学方程本身；运动学是窄向加成（专治 v-OOD 位置），且在外观 OOD 上要付代价。**要让运动学全面生效，仍需 PIWM 的 extrinsic 化 + 编码对外观鲁棒（量化/增广）。**

## 5. Pixel 确认 + 跨域迁移（结论翻正）

### 5.1 uniform pixel by-horizon（PRED PSNR dB↑，位置权重高的尺）

| | h1 | h8 | h16 | **h28** | **both-OOD** |
|---|---|---|---|---|---|
| baseline_fr（纯FR） | 25.55 | 21.05 | 21.86 | 22.09 | 20.41 |
| structpos_fr_pw30（承重） | 26.86 | 22.68 | 22.57 | 22.41 | 21.30 |
| **structcv_fr_pw100（承重+运动学）** | 26.11 | 22.34 | 22.74 | **23.34** | **21.67** |

**在像素尺上承重+运动学明显净超纯 FR**：长程 h28 **+1.25dB**、both-OOD **+1.26dB**。latent 聚合尺没看出来是因为被 190 维黑盒稀释——pixel 才是对的尺。

### 5.2 迁移 collision / parabola（latent nMSE↓，both-OOD | h28cos）

| 域 | baseline_fr | structpos_pw30(承重) | structdyn_areg_pw30(承重+运动学) |
|---|---|---|---|
| parabola | 0.313 \| 0.95 | 0.341 \| 0.98 | **0.262 \| 0.97** ✅ |
| collision | **0.393** \| 0.58 | 0.596 \| 0.56 ⚠️ | 0.523 \| 0.53 ⚠️ |

- **parabola**：承重+可学accel **both-OOD 0.313→0.262**（accel 学到常重力）✅；承重单独反而略差 → 这里**运动学是关键**。
- **collision**：承重和运动学都**变差**（冲量非光滑，smooth accel MLP + 硬承重都扛不住）。纯 FR 仍最好。

### 5.3 refined 结论

**运动学能帮 OOD/长程,但只在"光滑动力学"域,且要配承重编码 + 对的尺(pixel)**:
- uniform(a=0)、parabola(a=常重力)：**承重 + 运动学净超纯 FR**(pixel h28 +1.25dB / parabola both-OOD -0.05)。
- collision(冲量,a 非光滑)：smooth accel MLP + 硬承重都失败 → 需要**无固定形式**的约束。

## 6. 下一步：无固定形式的 consistency loss（冲 collision 缺口）

新加的 `loss.consistency`（约束预测 slot 速度=真实 proprio 速度，无固定 accel 形式，[train.py](../../le-wm/train.py) L122-152）正是为 collision 冲量设计——软物理约束、不用 smooth accel 头。冒烟已过（loss 有限在降）。

**验证归属**：用户本人在跑自己的 consistency A/B（`uniform_cons_A_base` = 结构化+pw30+cons0 基线臂 等），本报告不重复 launch，避免抢卡。待其结果落地后接入分析。

**假设**：consistency 在 collision 上应优于 accel-MLP 运动学（form-free、不被冲量打败）；在 uniform/parabola 上应匹配或接近承重+accel-MLP。

---

## 7. 阶段总结（pos_weight 探究部分，已完成）

**"怎么让运动学帮 OOD/长程"的答案**：
1. **前提是 free-rollout**（已设默认）——它是主升力。
2. **让物理编码承重**（`pos_weight≈30`）+ **运动学** → 在**光滑动力学域净超纯 FR**：uniform **pixel h28 +1.25dB / both-OOD +1.26dB**；parabola **latent both-OOD 0.313→0.262**。
3. **对的尺是 pixel**（位置权重高）——latent 192 维聚合会把 2 维物理增益稀释掉。
4. **collision（冲量）失败**：smooth accel MLP + 硬承重都扛不住 → 交给 form-free 的 consistency loss（用户在验证）。

**一句话**：运动学确实能帮 OOD/长程，但要 **free-rollout 打底 + 承重编码 + 光滑动力学域 + pixel 尺**四个条件同时满足；冲量域需换 form-free 约束。

---

## 4. 附：已落地的默认变更

- **free-rollout 设为默认**（[lewm.yaml](../../le-wm/config/train/lewm.yaml)）：`wm.free_rollout=true, num_preds=8, loader.batch_size=64`。这是三域验证的净提升。teacher-forced baseline 需显式 `wm.free_rollout=false wm.num_preds=1`。
- 新增 `loss.pos_weight`（默认 1.0=原行为）。
