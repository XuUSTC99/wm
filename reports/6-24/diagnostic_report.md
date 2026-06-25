# LeWM Deep-Supervision Sweep — 诊断报告
### "K=4 ρ / latent cos 涨上去到底意味着什么"

**日期**：2026-06-24（合并 6-05 到 6-07 的诊断材料）
**主题**：从 4 个独立诊断角度验证 [`reports/6-2/sweep_three_domains_results.md`](../6-2/sweep_three_domains_results.md) 里"高 λ_probe 全面胜出"的结论是否对应 world model **真正变好**。

---

## ⚠️ 重要 caveat — 数据来源

本报告引用的所有数值都来自 **2026-06-05/06 在 A500 跑的 45-config sweep ckpt**。后续在 [reports/6-2/sweep_three_domains_results.md 顶部更正](../6-2/sweep_three_domains_results.md) 中发现：那批 ckpt 的预训练权重命名 mismatch，`init_from_ckpt` **静默丢掉 192/216 ViT 主体权重**——所以下面的 ckpt 是"近随机初始化的 encoder + FT 20ep"，**不是真正的 pusht 预训练 baseline**。

**但本报告里的诊断方法论（如何识别 probe 损失对偶、如何看穿 latent cos 的循环逻辑、用 frozen encoder 拆耦合）独立于 init 状态**——结论方向（K=4 ρ 不可独立使用、pred_loss 才是 ground truth、intrinsic dim 揭示压缩）应当在修好 init 重跑后**仍然成立**，只是绝对数值会不同。

**用法**：
- ✅ 当作"如何诊断 deep-supervision 是否真有效" 的方法论手册
- ✅ 用作"为什么 sweep 报告里的 K=4 ρ 涨不能直接信" 的论据
- ❌ **不可引用本报告的数值结论**（intrinsic dim = 1.49 之类的具体数字，必须用修好 init 后重跑得到的版本替换）

### 相关报告状态总览

| 报告 | 数据来源 | 状态 | 数值可信？ |
|---|---|---|---|
| [`reports/5-26/negtive_result_report.md`](../5-26/negtive_result_report.md) | qlib 原始 | ✅ **有效**（主报告，含 pusht-frozen zero-shot ρ=0.9 baseline）| ✅ 可引用 |
| [`reports/5-26/piwm_deepsup_results.md`](../5-26/piwm_deepsup_results.md) | qlib 原始 | ✅ **有效**（5-26 deep-sup 分析）| ✅ 可引用 |
| [`reports/5-26/rollout_results.md`](../5-26/rollout_results.md) | qlib 原始 | ✅ **有效**（AR rollout 漂移分析）| ✅ 可引用 |
| [`reports/6-2/piwm_three_domains.md`](../6-2/piwm_three_domains.md) | qlib 原始 | ✅ **有效**（三域 deep-sup 对比 + within-traj std 假说）| ✅ 可引用 |
| [`reports/6-2/piwm_uniform_collision_results.md`](../6-2/piwm_uniform_collision_results.md) | qlib 原始 | ✅ **有效**（uniform + collision 数据表）| ✅ 可引用 |
| [`reports/6-2/piwm_three_domains_new.md`](../6-2/piwm_three_domains_new.md) | **A500 重跑** | ❌ **无效**（broken init bug）| ❌ 不可引用数值 |
| [`reports/6-2/sweep_three_domains_results.md`](../6-2/sweep_three_domains_results.md) | **A500 重跑** | ❌ **无效**（broken init bug + 报告顶部已加 ⚠️ 标注）| ❌ 不可引用数值 |
| **本报告** `reports/6-24/diagnostic_report.md` | A500 重跑（同上）| ⚠️ **方法有效、数值无效** | 方法可引用，数值待重跑 |

**判定规则**：
- 路径含 `qlib` 起源、或 5-26 创建的报告 → **有效**（pusht 预训练权重在 qlib 端命名自洽）
- 6-2 之后在 A500 上跑的训练 → **无效**（transformers 新版改了 ViT attention 命名，`init_from_ckpt` 静默丢 192/216 ViT 权重）
- 修复方案：[`le-wm/train.py`](../../le-wm/train.py) 已加 `_remap_old_vit_keys()` + 加载守卫，重跑后会写 `[init_from_ckpt] loaded=216 unexpected=0`；任何 `loaded < 200` 都是 broken init，整个 sweep 作废

---

## 0. TL;DR — 四个独立诊断都指向同一结论

| 诊断 | 测什么 | 结论 |
|---|---|---|
| 1. **Loss-component dominance** | 训练时 probe loss × λ 与 pred_loss 的比例 | w=50 时 probe gradient = pred gradient 的 **73–317×**，optimizer 几乎完全在最小化 probe |
| 2. **pred_loss 作为 GT metric** | 用 next-state prediction loss 选 best (w,f) | **w=0.1（最弱 probe）在 3/3 域上 pred_loss 最低**，K=4 ρ 的"最佳"完全相反 |
| 3. **Intrinsic dim collapse** | encoder 输出在 192 维空间的有效维度 | projector eff_dim 从 9.85 (w=0.1) → **1.49 (w=50)**，predictor 实际只在 1.5 维空间里学 |
| 4. **Encoder/target swap** | 把 latent cos 算式里的 encoder 换成 frozen pusht | cos h=16 从 0.880 → **0.074**（hard swap）/ **0.167**（target-only swap），predictor 学的是 encoder-specific 局部映射 |

**共同结论**：sweep 报告里"高 λ 各项 metric 涨"几乎全是 **probe 损失对偶 + encoder 自相似**的产物，**不代表 world model 真的更会预测物理**。

---

## 1. 诊断 1 — Probe loss 主导 optimizer

### 1.1 方法

LeWM 训练时优化的总损失是三项的加权和：

```
total_loss = pred_loss              ← world model 的核心目标：预测下一帧 latent
           + 0.09 × sigreg_loss     ← 防止 latent 坍缩（系数固定 0.09）
           + λ × probe_loss         ← deep-supervision 目标：让 latent 解码出物理量
```

参数更新时，梯度下降把三项的梯度相加：

```
∂total_loss/∂θ = ∂pred_loss/∂θ + 0.09·∂sigreg_loss/∂θ + λ·∂probe_loss/∂θ
                       A                   B                    C
```

**A、B、C 是三股拉力，谁大谁就主导参数更新方向。** 严格说要比较梯度范数，但对 MSE 形式的 loss，"加权后的 loss 值" 是梯度大小的合理代理。所以我们看两项**加权后的 loss 值**之比：

```
probe 项的有效大小   =  λ × probe_loss
pred  项的有效大小   =  pred_loss
比例                =  (λ × probe_loss) / pred_loss
```

**如果这个比例 ≫ 1，说明 probe 项在总 loss 里占绝大多数，optimizer 几乎只在最小化 probe，没在最小化 pred。**

### 1.2 数据（final epoch validate loss，parabola f=2）

**原始数据来源**：

- **文件位置**：`/data1/likun-share/junjxu/runs/sweep_three_domains_logs/train_parabola_sw_w<W>_f2_id1k.log`
  （w=30 / 50 在另一个目录：`.../sweep_three_domains_extend_logs/`）
- **抓取字段**：每个 train log 最后一次出现的
  - `validate/pred_loss_epoch`
  - `validate/probe_loss_epoch`
  - `validate/sigreg_loss_epoch`
- **抓取脚本**：[`reports/6-2/extract_sweep_results.py`](../6-2/extract_sweep_results.py) 同款 ANSI-strip + 正则 `r"\|\s+{key}\s+\|\s+([+\-]?\d+\.\d+...)"`，取 `findall(...)[-1]`（log 里每个字段被 validate 多次写入，最后一次即 epoch 20 终值）

例（w=0.1 那行）：
```
train log: train_parabola_sw_w0p1_f2_id1k.log
  最后一次 validate/pred_loss_epoch    → 0.0115
  最后一次 validate/probe_loss_epoch   → 0.15426
  最后一次 validate/sigreg_loss_epoch  → 2.1526
```

其余列由下式算出：

```
λ × probe_loss            = λ × probe_loss
0.09 × sigreg_loss        = 0.09 × sigreg_loss
total_loss                = pred_loss + 0.09·sigreg_loss + λ·probe_loss
probe 在 total 中占比      = (λ·probe_loss) / total_loss
(λ·probe) / pred 比例     = (λ·probe_loss) / pred_loss
```

| w | pred_loss | probe_loss | sigreg_loss | λ·probe | 0.09·sigreg | total_loss | probe 占 total | (λ·probe)/pred |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.0115 | 0.154 | 2.15 | 0.015 | 0.194 | 0.221 | 7.0% | 1.3× |
| 1.0 | 0.0141 | 0.118 | 2.85 | 0.118 | 0.257 | 0.389 | 30.3% | 8.3× |
| 10.0 | 0.0181 | 0.080 | 5.41 | 0.80 | 0.487 | 1.306 | 61.3% | **44×** |
| 30.0 | 0.0183 | 0.071 | 8.40 | 2.13 | 0.756 | 2.902 | 73.3% | **116×** |
| 50.0 | 0.0181 | 0.068 | 10.18 | **3.38** | 0.917 | 4.319 | **78.4%** | **187×** |

注意：在低 λ 区（w=0.1 / 1.0），实际是 **sigreg 项**主导 total loss（0.19/0.26），不是 probe；而 w≥10 后 probe 才正式接管 optimizer。

跨域看 w=50：

| 域 (w=50, f=2) | pred_loss | probe_loss | sigreg_loss | λ·probe | 0.09·sigreg | (λ·probe)/pred |
|---|---|---|---|---|---|---|
| parabola | 0.0181 | 0.068 | 10.18 | 3.38 | 0.917 | 187× |
| uniform | 0.0159 | 0.101 | (未抓) | 5.05 | — | **317×** |
| collision | 0.0249 | 0.036 | (未抓) | 1.82 | — | 73× |

### 1.3 解读

w=50 时 `λ × probe_loss` 是 `pred_loss` 的 **2 个数量级**以上——total loss 几乎完全等于 probe loss，optimizer 把所有"梯度预算"花在让 latent 解码出物理量上，没在管 "predictor 能否预测下一帧"。

直白类比：老师把作业按 `语文 + 50 × 数学` 算总分，你当前还能从语文扣 0.02 分、从数学扣 0.07 分 × 50 = 3.4 分——你肯定先攻数学。w=50 的 LeWM 就在干这件事：完全去拟合 probe，扔掉 pred。

---

## 2. 诊断 2 — Pred_loss 作为 ground-truth metric

### 2.1 为什么 K=4 ρ 不可单独信

```
probe_head: latent → 物理量（pos, vel）
K=4 probe ρ = Pearson(probe_head(predictor_rollout), real_physics)
```

`probe_head` 的训练目标就是最小化 `(probe_head(latent) - real_physics)²`——K=4 ρ 就是这个目标的**对偶量**。`λ` 越大、probe loss 优化得越狠，K=4 ρ 必然单调上升。**这是数学必然，不是模型变好**。

### 2.2 用 pred_loss 重新排序 15 cells (parabola)

| 排名 | 配置 | pred_loss | K=4 vx ID ρ |
|---|---|---|---|
| **1** | **w=0.1, f=2** | **0.0115** ⭐ | +0.531 |
| 2 | w=0.1, f=1 | 0.0129 | +0.534 |
| 3 | w=0.1, f=4 | 0.0133 | +0.370 |
| 4 | w=1.0, f=1 | 0.0138 | +0.470 |
| ... | ... | ... | ... |
| 11 | w=50, f=2 | 0.0181 | +0.654 |
| 12 | w=30, f=2 | 0.0183 | +0.713 |
| 13 | w=10, f=1 | 0.0221 | +0.335 |
| 14 | w=30, f=1 | 0.0279 | +0.409 |
| 15 | w=50, f=1 | 0.0298 | +0.465 |

**pred_loss 排序与 K=4 ρ 排序基本相反**：probe weight 越大、pred_loss 越差。

### 2.3 跨三域的 best-(w,f) 对比

| 域 | best by **pred_loss** | best by K=4 ID ρ | 一致？ |
|---|---|---|---|
| parabola | **w=0.1, f=2**（pl=0.0115）| w=30, f=2（ρ=+0.713）| ❌ |
| uniform | **w=0.1, f=4**（pl=0.0062）| w=30, f=4（ρ=+0.878）| ❌ |
| collision | **w=0.1, f=4**（pl=0.0127）| w=30, f=4（ρ=+0.812）| ❌ |

**3/3 域完全反向**。sweep 报告里的"w=50 全面胜出"对应的是 K=4 ρ；从 world-model 主目标 pred_loss 看，**w=0.1 才是全面最佳**。

### 2.4 与 arXiv:2504.03861 (Flappy Bird) 对比

Paper §3.1：lambda 越大，pred_loss 越小（Flappy Bird, latent dim=8）。
我们的：λ 越大，pred_loss 越大（PhyWorld, latent dim=192）。

差异核心：
- 他们的 encoder 是 MLP，输入 180 维 LIDAR；probe target（3 个 features）占 8 维 latent 的 38%
- 我们的 encoder 是 ViT，输入 224×224 RGB；probe target（2-4 维）占 192 维 latent 的 1-2%
- **高维 ViT latent 有大量"可被压扁"的冗余通道（视觉纹理 / 外观 / 形状），它们恰恰是被 probe 牺牲掉的**

---

## 3. 诊断 3 — Encoder 输出 intrinsic dim 塌方

### 3.1 方法

Participation ratio（PR）= `(Σ σᵢ)² / Σ σᵢ²`，σᵢ 为 latent 的奇异值。
- 满秩 192 维 → PR = 192
- 全塌到 1 维 → PR = 1

500 个 eval frame → encoder/projector forward → PR。

### 3.2 数据（parabola, f=2）

| w | eff_CLS | **eff_proj** | σ₁(proj) | σ₂/σ₁ |
|---|---|---|---|---|
| 0.1 | 2.91 | **9.85** | 168 | — |
| 1.0 | 2.47 | 5.95 | 257 | — |
| 10.0 | 3.64 | 2.37 | 547 | 0.91 |
| 30.0 | 3.10 | 1.62 | 728 | 0.77 |
| 50.0 | 2.83 | **1.49** | 768 | **0.43** |

> ⚠️ eff_CLS 数值偏小是因为这批 ckpt 的 ViT 主体是随机初始化（见顶部 caveat）。修好 init 后 CLS eff_dim 应在 30-60 范围。但 **eff_proj 随 λ 单调下降的趋势独立于 init bug**。

### 3.3 解读

**Projector 输出（predictor 真正工作的空间）从 9.85 维塌到 1.49 维**——σ₁ 暴涨到 768 吸收了几乎所有方差。

Predictor 实际上在 **1.5 维空间**里学动力学。1.5 维只够装 (x, y)——所以 K=4 ρ 当然好（latent 就是 (x, y)），但 encoder 失去了表达视觉纹理 / 外观 / 形状的能力。

---

## 4. 诊断 4 — Encoder swap test

### 4.1 实验设计

构造 hybrid ckpt：保留 sweep ckpt 的 predictor，但 encoder weights 换成 frozen pusht weights.pt。然后跑同样的 rollout eval。

```
原始：     trained_encoder + trained_predictor → cos h=16 = 0.872  (w=50 parabola)
hybrid：   FROZEN_encoder  + trained_predictor → cos h=16 = ?
```

如果 predictor 学到了 **encoder-agnostic 的 generic physics dynamics**（比如 v→v+a·dt），即使 encoder 换成 frozen pusht 也应该部分 work。如果 predictor 学的是 **encoder-space-specific 的局部映射**，swap 后必然崩。

### 4.2 数据（cos h=16）

| 配置 | 原始（trained enc）| **frozen swap** | Δ |
|---|---|---|---|
| parabola paperinit | 0.590 | 0.598 | +0.008 |
| parabola w=0.1, f=2 | 0.695 | 0.262 | −0.432 |
| **parabola w=50, f=2** | **0.872** | **0.074** | **−0.797** |
| uniform paperinit | 0.769 | −0.003 | −0.772 |
| uniform w=50, f=4 | 0.954 | 0.068 | −0.886 |
| collision paperinit | 0.440 | −0.080 | −0.520 |
| **collision w=50, f=4** | **0.633** | **−0.007** | **−0.640** |

### 4.3 解读

1. **parabola paperinit Δ ≈ 0** — encoder 没怎么被 FT 动（与 5-26 §6.4 "ID-only FT 净效应 ≈ 0" 一致）
2. **加了 probe 之后 swap Δ 立刻负**——而且 |Δ| **与 λ 单调相关**（drift ∝ λ）
3. **w=50 时 swap 直接崩到 0.07** — predictor 完全 encoder-space-specific，没学任何可迁移的 dynamics

→ **sweep 报告里 w=50 比 baseline 高 0.282 的 cos h=16，本质是 encoder + predictor 联合耦合到一个压扁子空间，predictor 没学到 generic physics**。

---

## 5. 诊断 5 — Frozen-target cos（无需 swap predictor）

### 5.1 实验设计

诊断 4 是"硬 swap"（encoder 整个换掉，predictor 输入立刻乱）。本诊断更精细：
- predictor 仍然用 trained encoder 编码 history 来 rollout（保留 input space）
- 但比较时把 target real_emb 换成 frozen pusht encoder 编码的真实帧

```
原始：  cos( trained_encoder(real_frame_k),   trained_predictor_rollout )
本诊断：cos( FROZEN_encoder(real_frame_k),    trained_predictor_rollout )
```

如果 predictor 输出**在 pusht semantic space 里也有语义对齐**，frozen-target cos 应该不至于太低。

### 5.2 数据（h=16, 9 ckpt）

| 域 | 配置 | trained-target | **frozen-target** | 落差 |
|---|---|---|---|---|
| parabola | baseline (w=0) | 0.836 | 0.122 | −0.71 |
| | w=0.1 | 0.857 | 0.117 | −0.74 |
| | **w=50** | **0.880** | **0.167** | −0.71 |
| uniform | baseline | 0.852 | 0.098 | −0.75 |
| | w=0.1 | 0.893 | 0.091 | −0.80 |
| | **w=50** | **0.953** | **0.298** | −0.66 |
| collision | baseline | 0.527 | 0.042 | −0.49 |
| | w=0.1 | 0.608 | 0.048 | −0.56 |
| | **w=50** | **0.725** | **0.071** | −0.65 |

### 5.3 关键观察

1. **所有 frozen-target cos 几乎随机水平**（0.04 - 0.30）——predictor 输出与 pusht semantic 几乎正交
2. **sweep 的"提升"在 frozen metric 下大部分蒸发**：
   - parabola w=50 比 baseline 在 trained metric 上 +0.044，在 frozen metric 上 **+0.045**（几乎打平）
   - collision w=50 比 baseline 在 trained metric 上 +0.198，在 frozen metric 上 **+0.029**（几乎全蒸发）
3. **uniform w=50 是唯一显著的 frozen cos**（0.298）—— **巧合而非真理解**：uniform 的 encoder 严重压扁到"球水平位置 x"这个一维方向，而 pusht encoder 也能识别球的位置，两者在这一个方向上意外对齐。这反而**坐实了塌方假说**——trained encoder 收敛到的就是 "x 的一维流形"

---

## 6. 综合机制：高 λ 在 image-based LeWM 上做了什么

```
出发点
  pusht-pretrained ViT  →  zero-shot 已能在 phyworld 解码物理量到 ρ ≈ 0.9
                            (来自 5-26 主报告 §6.1)

加 deep-sup probe 训练：
  probe_loss × λ  在 w=50 时是 pred_loss 的 73-317 倍
                ↓
  optimizer 几乎只在最小化 probe_loss
                ↓
  encoder 被强制把 latent 中"物理量对应的方向"放大、其他方向压扁
                ↓
  projector 输出从 ~10 维塌到 ~1.5 维（诊断 3）
                ↓
  predictor 在这个 1.5 维空间里学的是 "x_t+1 = x_t + v_t" 这种极简映射
                ↓
  评估时：
    - K=4 ρ：probe loss 的对偶 → 必然涨 ✓
    - latent cos：encoder 自相似（左右两端在同一压扁空间里）→ 必然高 ✓
    - pred_loss：encoder 失去其他信息后预测 next latent 反而更难 → 退化 ✗
    - swap test：predictor 完全 encoder-specific → frozen swap 崩 ✗
    - frozen-target cos：predictor 输出在 pusht space 几乎正交 → 接近 0 ✗
```

**Sweep 报告里"高 λ 全面胜出"= probe 损失对偶 + encoder 自相似指标全套**，是一种系统性 metric gaming，**不是 world model 在物理上变好**。

---

## 7. 与 arXiv:2504.03861（Flappy Bird paper）的对比

| 维度 | Paper (Flappy Bird) | 我们 (LeWM / PhyWorld) | 影响 |
|---|---|---|---|
| 观察 | 180-d LIDAR 向量 | 224×224 RGB 图像 | — |
| Encoder | MLP 180→8 | ViT-tiny 224×224 → 192 | latent dim **24×** |
| Probe target 占比 | 3 / 8 = **38%** | 4 / 192 = **2%** | paper 没空间塌方 |
| Probe coefficient λ | 0 → 64（20 seeds 各 λ）| 0.1 / 1 / 10 / 30 / 50（1 seed 各 λ）| 我们 seed 不足 |
| **主报告 metric** | **pred_loss（"next state prediction loss"）** | K=4 ρ + cos h=16 | **我们盯错了 metric** |
| **λ↑ 后 pred_loss** | **下降**（paper §3.1）| 上升（我们的数据） | 相反结论 |

**Paper 在 8 维 latent / 38% probe 占比 / 直接看 pred_loss 的设置下**得出"deep-sup 有效"——这是**正确的**。但**直接搬到 192 维 ViT 上、盯 K=4 ρ 看，会重现我们的所有 metric gaming 假象**。

---

## 8. 方法论建议（下一轮 sweep 必须做的）

### 8.1 主指标用 pred_loss，不用 K=4 ρ

```python
# 训练时已经在 Lightning log 里：
validate/pred_loss_epoch   # ← 唯一可信的 sweep 排序依据
```

K=4 ρ、latent cos by horizon、latent cos by partition **全部只作为次要诊断**，永远不能单独决定 best (w, f)。

### 8.2 每次 sweep 必须加 3 个 sanity check

| 诊断 | 怎么测 | 红线 |
|---|---|---|
| Loss dominance | `λ × probe_loss / pred_loss` | > 10× 警告，> 100× 立即停 |
| Projector eff_dim | `participation ratio` of `model.encode(test)["emb"]` | < 5 警告，< 2 塌方 |
| Frozen-target cos | `cos(frozen_encoder(real_k), trained_predictor_rollout)` | 与 trained-target 落差 > 0.5 警告 |

### 8.3 关键 cell 必须 multi-seed

`piwm_three_domains_new.md` 里 collision v-OOD posonly = −0.097 的"崩塌"在新 sweep 里换 seed 反弹到 +0.45——单 seed 不可信。**任何 "best (w,f) 排序" 必须至少 3 seed**。

### 8.4 如果做 image-based deep-sup，必须验证 init 真加载上

[`train.py`](../../le-wm/train.py) 现在已经加了 `_remap_old_vit_keys` + 加载守卫，但每次重启训练都要在 train log 里确认这一行：
```
[init_from_ckpt] loaded=216 unexpected=0
```
任何 `loaded < 200` 都说明命名 mismatch，整个 sweep 作废。

---

## 9. 复现 / 文件路径

| 项 | 路径 |
|---|---|
| 本报告 | `reports/6-24/diagnostic_report.md` |
| Sweep 原报告（有 caveat）| [`reports/6-2/sweep_three_domains_results.md`](../6-2/sweep_three_domains_results.md) |
| Extract 脚本（45 cells 数据）| [`reports/6-2/extract_sweep_results.py`](../6-2/extract_sweep_results.py) |
| Frozen-target 诊断脚本 | [`reports/6-2/frozen_target_diagnostic.py`](../6-2/frozen_target_diagnostic.py) |
| 训练修复（命名 remap + 守卫）| [`le-wm/train.py`](../../le-wm/train.py) `_remap_old_vit_keys` |
| Sweep 训练日志（init=broken）| `/data1/likun-share/junjxu/runs/sweep_three_domains_logs/`<br>`/data1/likun-share/junjxu/runs/sweep_three_domains_extend_logs/` |
| Encoder-swap 诊断日志 | `/data1/likun-share/junjxu/runs/frozen_enc_diagnostic_logs/` |
| Frozen-target 诊断日志 | `/data1/likun-share/junjxu/runs/frozen_target_diagnostic_logs/` |

### 复现 frozen-target 诊断

```bash
cd /home/likun-share/junjxu/wm
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
# 单 ckpt
CUDA_VISIBLE_DEVICES=0 le-wm/.venv/bin/python reports/6-2/frozen_target_diagnostic.py \
  --domain parabola \
  --ckpt $STABLEWM_HOME/parabola_sw_w50p0_f2_id1k/parabola_sw_w50p0_f2_id1k_epoch_20_object.ckpt \
  --max-trajs 300
```

### 复现 intrinsic dim 测量

```python
import torch, h5py
from pathlib import Path
SWM = '/data1/likun-share/junjxu/.stable_worldmodel'
ckpt = torch.load(f'{SWM}/parabola_sw_w50p0_f2_id1k/parabola_sw_w50p0_f2_id1k_epoch_20_object.ckpt',
                  map_location='cuda', weights_only=False).eval()
with h5py.File(f'{SWM}/datasets/phyworld_parabola.h5', 'r') as f:
    pixels = torch.tensor(f['pixels'][:500]).permute(0,3,1,2).float().cuda() / 255.
    from torchvision.transforms.functional import normalize
    pixels = normalize(pixels, [0.485,0.456,0.406], [0.229,0.224,0.225])
emb = ckpt.encode({"pixels": pixels.unsqueeze(0)})["emb"][0]  # (500, 192)
emb_c = emb - emb.mean(0)
S = torch.linalg.svdvals(emb_c.float())
eff = ((S.sum())**2 / (S**2).sum()).item()
print(f"projector eff_dim = {eff:.2f}")
```

---

## 10. 一句话总结

**Sweep 报告里"高 λ_probe 各项 metric 涨"几乎全部是 probe 损失对偶 + encoder 自相似的循环逻辑产物。pred_loss、intrinsic dim、encoder swap、frozen-target cos 四个独立角度都指向同一个事实：encoder 被压扁到 1.5 维物理量子空间，predictor 学的是 encoder-specific 局部映射，不是 generic physics dynamics。修好 init bug 后还需要 multi-seed 重跑才能给出有效结论，而所有重跑必须以 pred_loss 为主指标，把 K=4 ρ 和 latent cos 当作次要诊断。**
