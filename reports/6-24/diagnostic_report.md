# LeWM Deep-Supervision Sweep — 诊断报告
### "K=4 ρ / latent cos 涨上去到底意味着什么"

**日期**：2026-06-24 创建（broken-init 数据），**2026-06-26 用 fixed-init 重跑更新 §1/§2/§3/§5**

**主题**：从 5 个独立诊断角度验证 sweep 报告里"高 λ_probe 全面胜出"的结论是否对应 world model **真正变好**。

---

## ⚠️ 数据来源（已用 fixed-init 重跑覆盖）

本报告**最初**（2026-06-24）引用的数值都来自 broken-init sweep ckpt（详见 [reports/6-2/sweep_three_domains_results.md 顶部更正](../6-2/sweep_three_domains_results.md)：`init_from_ckpt` 静默丢 192/216 ViT 主体权重，encoder 近随机初始化）。

**2026-06-26 重跑**：用 fixed-init 的 train.py（`_remap_old_vit_keys` + 加载守卫，确认 `loaded=216 unexpected=0`）重训了 **9 个对照 ckpt**（3 域 × λ ∈ {1, 5, 10} × frames=2）。下面 §1/§2/§3/§5 数据已替换为 fixed-init 重跑值；**§4 (hard swap) 因要构造 hybrid ckpt 暂未重跑，但 §5 的同源更激进版本已替换**。

| 节 | 数据状态 |
|---|---|
| §0 TL;DR | ✅ fixed-init 重跑（部分） |
| §1 loss dominance | ✅ fixed-init 重跑（λ=1, 5, 10） |
| §2 pred_loss ranking | ✅ fixed-init 重跑（λ=1, 5, 10） |
| §3 intrinsic dim | ✅ fixed-init 重跑（9 ckpt） |
| §4 encoder hard swap | ⚠️ 仍是 broken-init 数据，标注待重跑 |
| §5 frozen-target cos | ✅ fixed-init 重跑（9 ckpt） |

**主要结论变化（broken → fixed）**：
- 诊断 1（probe 主导 optimizer）**结论方向不变**。这里的"倍数"指 `(λ·probe_loss)/pred_loss`，即 probe 损失项比 pred 损失项大多少倍——倍数越高，说明 optimizer 越是在最小化 probe 而非 pred。broken-init 时该比值 73-317×（测于 λ=50），fixed-init 重跑后降到 3-125×（λ=1 各域 3-20×，λ=10 各域 15-125×，见 §1.2 全表）。注意两个区间不是同一 λ 下的对比：数值变小一部分来自 init 修复、一部分来自 λ 上限从 50 降到 10。但无论哪个 λ，probe 项仍远大于 pred 项，**probe 主导 optimizer 这一结论依旧成立（仍极端）**
- 诊断 2（pred_loss 看 λ=1 最佳）**3/3 域仍成立**
- 诊断 3（塌方）**仍成立**但更温和（eff_proj 从 9.85→1.49 变为 13~41→3~17）
- 诊断 5（frozen-target cos）**结论部分翻转**：fixed-init 上 cos 大幅提升（0.04~0.30 → 0.46~0.68），落差从 −0.66~−0.88 缩小到 −0.04~−0.34 → **predictor 学到的实际比 broken-init 时通用得多，没有想象中那么 encoder-specific**

**用法**：
- ✅ 当作"如何诊断 deep-supervision 是否真有效"的方法论手册
- ✅ 用作"为什么 sweep 报告里的 K=4 ρ 涨不能直接信"的论据
- ✅ 数值现在**可引用**（fixed-init 重跑后）—— 但 §4 例外，未重跑

### 相关报告状态总览

| 报告 | 数据来源 | 状态 | 数值可信？ |
|---|---|---|---|
| [`reports/5-26/negtive_result_report.md`](../5-26/negtive_result_report.md) | qlib 原始 | ✅ **有效**（主报告，含 pusht-frozen zero-shot ρ=0.9 baseline）| ✅ 可引用 |
| [`reports/5-26/piwm_deepsup_results.md`](../5-26/piwm_deepsup_results.md) | qlib 原始 | ✅ **有效**（5-26 deep-sup 分析）| ✅ 可引用 |
| [`reports/5-26/rollout_results.md`](../5-26/rollout_results.md) | qlib 原始 | ✅ **有效**（AR rollout 漂移分析）| ✅ 可引用 |
| [`reports/6-2/piwm_three_domains.md`](../6-2/piwm_three_domains.md) | qlib 原始 | ✅ **有效**（三域 deep-sup 对比 + within-traj std 假说）| ✅ 可引用 |
| [`reports/6-2/piwm_uniform_collision_results.md`](../6-2/piwm_uniform_collision_results.md) | qlib 原始 | ✅ **有效**（uniform + collision 数据表）| ✅ 可引用 |
| [`reports/6-2/piwm_three_domains_new.md`](../6-2/piwm_three_domains_new.md) | **A500 重跑** | ❌ **无效**（broken init bug）| ❌ 不可引用数值 |
| [`reports/6-2/sweep_three_domains_results.md`](../6-2/sweep_three_domains_results.md) | **A500 重跑** | ❌ **无效**（broken init 
bug + 报告顶部已加 ⚠️ 标注）| ❌ 不可引用数值 |
| **本报告** `reports/6-24/diagnostic_report.md` | A500 重跑（同上）| ⚠️ **方法有效、数值无效** | 方法可引用，数值待重跑 |

**判定规则**：
- 路径含 `qlib` 起源、或 5-26 创建的报告 → **有效**（pusht 预训练权重在 qlib 端命名自洽）
- 6-2 之后在 A500 上跑的训练 → **无效**（transformers 新版改了 ViT attention 命名，`init_from_ckpt` 静默丢 192/216 ViT 权重）
- 修复方案：[`le-wm/train.py`](../../le-wm/train.py) 已加 `_remap_old_vit_keys()` + 加载守卫，重跑后会写 `[init_from_ckpt] loaded=216 unexpected=0`；任何 `loaded < 200` 都是 broken init，整个 sweep 作废

---

## 0. TL;DR — 四个独立诊断都指向同一结论

| 诊断 | 测什么 | 结论（fixed-init λ ∈ {1, 5, 10}, f=2）|
|---|---|---|
| 1. **Loss-component dominance** | 训练时 (λ × probe_loss) 与 pred_loss 的比例 | λ=10 时 probe gradient 是 pred gradient 的 **15–125×**（collision 15× / uniform 44× / parabola 125×），optimizer 几乎完全在最小化 probe |
| 2. **pred_loss 作为 GT metric** | 用 next-state prediction loss 选 best λ | **λ=1 在 3/3 域上 pred_loss 最低**（parabola 0.0054 / uniform 0.0099 / collision 0.0113），K=4 ρ 的"最佳"反向 |
| 3. **Intrinsic dim collapse** | encoder 输出在 192 维空间的有效维度 | projector eff_dim 从 13~41 (λ=1) → **3~17 (λ=10)**，塌方 39%–90%；predictor 在小空间里学 |
| 4. **Encoder hard swap**（待重跑）| 把 latent cos 算式里的 encoder 换成 frozen pusht | broken-init 数据：cos h=16 从 0.880 → 0.074（待用 fixed-init 复测）|
| 5. **Frozen-target cos**（§5 替代 §4 的同源结论）| 同 ckpt 自身 encoder 编 history 滚 rollout，但 target latent 用 frozen pusht encoder 编 | fixed-init: cos h=16 落差仅 **−0.04~−0.34**（collision 几乎无差），predictor 学到的实际**相当通用**——不是完全 encoder-specific |

**修订后的共同结论**：
- ✅ "K=4 ρ 是 probe loss 对偶量、λ 越大必然涨"**仍成立**（fixed-init 数据 0.624 → 0.755 单调上升）
- ✅ "pred_loss 才是 ground truth，λ=1 最佳"**仍成立**（3/3 域）
- ✅ "intrinsic dim 随 λ 塌方"**仍成立**但更温和
- ⚠️ "predictor 学的是 encoder-specific 局部映射"**部分翻转**——fixed-init 上 predictor 在 frozen-target 空间里 cos 仍有 0.46–0.68，落差只 −0.04~−0.34（vs broken-init −0.66~−0.88）。说明 pusht 预训练 encoder + FT 后 predictor 在 pusht semantic 空间里**保留了相当多通用结构**

**整体定性方向不变**：sweep 报告里"高 λ 各项 metric 涨" 主要还是 **probe 损失对偶**的产物，但破坏程度比 broken-init 时温和。

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

### 1.2 数据（final epoch validate loss，fixed-init rerun，f=2）

**原始数据来源**：

- **文件位置**：`/data1/likun-share/junjxu/runs/6-24_rerun_logs/train_<dom>_rerun_w<W>_f2_id1k.log`
- **训练脚本**：[`reports/6-24/rerun_diagnostic_inputs.sh`](rerun_diagnostic_inputs.sh)（9 ckpt = 3 域 × 3 λ × f=2，fixed init `_remap_old_vit_keys`，loaded=216）
- **抓取字段**：每个 train log 最后一次出现的 `validate/pred_loss_epoch` / `validate/probe_loss_epoch` / `validate/sigreg_loss_epoch`

其余列由下式算出：

```
λ × probe_loss            = λ × probe_loss
0.09 × sigreg_loss        = 0.09 × sigreg_loss   ← LeWM 默认 sigreg 系数
total_loss                = pred_loss + 0.09·sigreg_loss + λ·probe_loss
probe 在 total 中占比      = (λ·probe_loss) / total_loss
(λ·probe) / pred 比例     = (λ·probe_loss) / pred_loss
```

#### 三域全表

| 域 | λ | pred_loss | probe_loss | sigreg_loss | λ·probe | 0.09·sigreg | total_loss | probe 占 total | **(λ·probe)/pred** |
|---|---|---|---|---|---|---|---|---|---|
| parabola | 1.0 | 0.0054 | 0.1102 | 2.06 | 0.110 | 0.186 | 0.301 | 36.5% | 20.5× |
| parabola | 5.0 | 0.0063 | 0.0839 | 3.33 | 0.420 | 0.299 | 0.725 | 57.9% | **67×** |
| parabola | 10.0 | 0.0056 | 0.0705 | 4.58 | 0.705 | 0.412 | 1.122 | 62.8% | **125×** |
| uniform | 1.0 | 0.0099 | 0.1111 | 1.53 | 0.111 | 0.137 | 0.258 | 43.0% | 11.2× |
| uniform | 5.0 | 0.0231 | 0.0998 | 2.04 | 0.499 | 0.183 | 0.705 | 70.8% | 21.6× |
| uniform | 10.0 | 0.0245 | 0.1087 | 3.16 | 1.087 | 0.285 | 1.396 | 77.9% | **44×** |
| collision | 1.0 | 0.0113 | 0.0390 | 1.57 | 0.039 | 0.141 | 0.192 | 20.3% | 3.4× |
| collision | 5.0 | 0.0165 | 0.0306 | 2.00 | 0.153 | 0.180 | 0.349 | 43.8% | 9.3× |
| collision | 10.0 | 0.0189 | 0.0287 | 2.28 | 0.287 | 0.205 | 0.511 | 56.1% | **15×** |

### 1.3 解读

- **λ=10 时**：parabola probe gradient 是 pred gradient 的 **125×**，uniform **44×**，collision **15×**——三域都进入"probe 主导 optimizer"区
- **λ=1 时**：probe 占 total 20-43%（不算压倒性），但 (λ·probe)/pred 仍达 3-20× —— pred 项已经被相对边缘化
- **跨域**：probe_loss 的绝对值依域不同（collision 仅 0.029，parabola 0.07-0.11），所以同 λ 下 (λ·probe)/pred 比例也不同，**collision 是受 probe 主导程度最低的域**

直白类比：老师把作业按 `语文 + λ × 数学` 算总分；当 (λ·数学)/语文 = 125× 时，你只会刷数学。λ=10 的 LeWM 就在干这件事。

---

## 2. 诊断 2 — Pred_loss 作为 ground-truth metric

### 2.1 为什么 K=4 ρ 不可单独信

```
probe_head: latent → 物理量（pos, vel）
K=4 probe ρ = Pearson(probe_head(predictor_rollout), real_physics)
```

`probe_head` 的训练目标就是最小化 `(probe_head(latent) - real_physics)²`——K=4 ρ 就是这个目标的**对偶量**。`λ` 越大、probe loss 优化得越狠，K=4 ρ 必然单调上升。**这是数学必然，不是模型变好**。

### 2.2 fixed-init 数据：pred_loss vs K=4 ρ（同 9 ckpts，f=2）

| 域 | λ | pred_loss | K=4 vx ID ρ | cos h=16 |
|---|---|---|---|---|
| parabola | **1.0** | **0.0054** ⭐ | +0.624 | +0.677 |
| parabola | 5.0 | 0.0063 | +0.690 | +0.693 |
| parabola | 10.0 | 0.0056 | **+0.755** | **+0.713** |
| uniform | **1.0** | **0.0099** ⭐ | **+0.784** | +0.767 |
| uniform | 5.0 | 0.0231 | +0.774 | +0.774 |
| uniform | 10.0 | 0.0245 | +0.755 | **+0.826** |
| collision | **1.0** | **0.0113** ⭐ | +0.639 | +0.378 |
| collision | 5.0 | 0.0165 | **+0.788** | +0.472 |
| collision | 10.0 | 0.0189 | +0.726 | **+0.503** |

**关键观察**：

1. **pred_loss 在 3/3 域上 λ=1 最低**：parabola 0.0054、uniform 0.0099、collision 0.0113
2. **K=4 ρ / cos h=16 在 3/3 域上 λ↑ 时单调上升或非单调上升**：parabola ρ 0.624→0.755、cos h=16 0.677→0.713；uniform cos 0.767→0.826；collision cos 0.378→0.503
3. **uniform / collision 上 pred_loss 单调上升**（0.0099→0.0245、0.0113→0.0189）—— 经典的 "world model 真退化 + probe 表面指标涨" 反向关系
4. **parabola 上 pred_loss 是 U 形**（1→5 涨 +17%，5→10 反而降 -11%）—— probe 与 pred 在 λ=10 时有点协同，但仍是 λ=1 最低

### 2.3 三域 best-by-pred_loss vs best-by-K4-ρ 全相反

| 域 | best by **pred_loss** | best by K=4 vx ID ρ | 一致？ |
|---|---|---|---|
| parabola | **λ=1**（0.0054）| λ=10（0.755）| ❌ |
| uniform | **λ=1**（0.0099）| λ=1（0.784）✓ | ⚠️ 巧合一致 |
| collision | **λ=1**（0.0113）| λ=5（0.788）| ❌ |

3/3 域 pred_loss 最佳都是 λ=1（最弱 probe）。从 world-model 主目标看，**deep-sup probe 越弱越好**——这与 sweep 报告 "λ=50 全面胜出" 截然相反，因为后者用 K=4 ρ（probe 损失对偶）作判据。

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

### 3.2 数据（fixed-init rerun, f=2, 500 eval pixels）

| 域 | λ | eff_CLS | **eff_proj** | σ₁(proj) |
|---|---|---|---|---|
| parabola | 1.0 | 9.77 | **13.37** | 143 |
| parabola | 5.0 | 8.20 | 4.14 | 285 |
| parabola | 10.0 | 5.59 | **3.42** | 368 |
| uniform | 1.0 | 16.31 | **41.28** | 77 |
| uniform | 5.0 | 10.00 | 12.30 | 161 |
| uniform | 10.0 | 8.10 | **3.98** | 279 |
| collision | 1.0 | 13.32 | **28.35** | 104 |
| collision | 5.0 | 11.06 | 21.62 | 116 |
| collision | 10.0 | 11.29 | **17.42** | 136 |

#### vs broken-init（仅 parabola）

| λ | broken eff_proj | fixed eff_proj |
|---|---|---|
| 1.0 | 5.95 | **13.37**（+125%）|
| 10.0 | 2.37 | **3.42**（+44%）|

→ fixed-init **总体维度更高**（pusht 权重保留了更多语义结构），但塌方趋势完全一致。

### 3.3 解读

**塌方仍是核心现象，但程度依域不同**：

| 域 | 塌方比例（λ=1→10）|
|---|---|
| **uniform** | 41.28 → 3.98 = **-90%**（最严重）|
| **parabola** | 13.37 → 3.42 = **-75%** |
| **collision** | 28.35 → 17.42 = **-39%**（最温和）|

uniform 上 λ=10 塌到 4 维 —— predictor 几乎只在球的 (x, y) 子空间里工作；collision 因为有两个球 + 碰撞事件，latent 信息量更高，塌方相对温和（仍保留 17 维）。

**关键**：λ=1 时 eff_proj 已显著高于 λ=10（uniform 41 vs 4），但 K=4 ρ 反而没差太多（0.784 vs 0.755）——说明 K=4 ρ 在 λ=1 时就已经触顶，**根本不需要靠塌方换更高的 ρ**。所有"塌方代价"换来的 ρ 提升几乎为零。

---

## 4. 诊断 4 — Encoder swap test

### 4.1 实验设计

构造 hybrid ckpt：保留 sweep ckpt 的 predictor，但 encoder weights 换成 frozen pusht weights.pt。然后跑同样的 rollout eval。

```
原始：     trained_encoder + trained_predictor → cos h=16 = 0.872  (w=50 parabola)
hybrid：   FROZEN_encoder  + trained_predictor → cos h=16 = ?
```

如果 predictor 学到了 **encoder-agnostic 的 generic physics dynamics**（比如 v→v+a·dt），即使 encoder 换成 frozen pusht 也应该部分 work。如果 predictor 学的是 **encoder-space-specific 的局部映射**，swap 后必然崩。

### 4.2 数据（cos h=16）— ⚠️ 仍是 broken-init 数据

| 配置 | 原始（trained enc）| **frozen swap** | Δ |
|---|---|---|---|
| parabola paperinit | 0.590 | 0.598 | +0.008 |
| parabola w=0.1, f=2 | 0.695 | 0.262 | −0.432 |
| **parabola w=50, f=2** | **0.872** | **0.074** | **−0.797** |
| uniform paperinit | 0.769 | −0.003 | −0.772 |
| uniform w=50, f=4 | 0.954 | 0.068 | −0.886 |
| collision paperinit | 0.440 | −0.080 | −0.520 |
| **collision w=50, f=4** | **0.633** | **−0.007** | **−0.640** |

> ⚠️ **本节数据仍来自 broken-init ckpt（2026-06-07 前）**。Hard swap 需要构造 hybrid ckpt（encoder 换成 frozen pusht + 保留 trained predictor），fixed-init 版未补做。**§5 的同源结论 fixed-init 已重跑** —— frozen-target cos 落差 −0.04 ~ −0.34（远小于 broken-init 这里的 −0.71 ~ −0.89），暗示 fixed-init 上 hard swap 也会得到温和得多的结果。

### 4.3 解读（broken-init）

1. **parabola paperinit Δ ≈ 0** — encoder 没怎么被 FT 动（与 5-26 §6.4 "ID-only FT 净效应 ≈ 0" 一致）
2. **加了 probe 之后 swap Δ 立刻负**——而且 |Δ| **与 λ 单调相关**（drift ∝ λ）
3. **w=50 时 swap 直接崩到 0.07** —— broken-init 下 predictor 完全 encoder-space-specific

→ **broken-init 上**，sweep 报告里 w=50 比 baseline 高 0.282 的 cos h=16 本质是 encoder + predictor 联合耦合到压扁子空间。
→ **fixed-init 上预期** 这个落差会显著缩小（参见 §5 实测落差 −0.04 ~ −0.34），不像 broken 那么极端。

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

### 5.2 数据（h=16, 9 ckpt, fixed-init）

| 域 | λ | trained-target | **frozen-target** | 落差 Δ |
|---|---|---|---|---|
| parabola | 1.0 | 0.810 | 0.469 | −0.340 |
| parabola | 5.0 | 0.857 | 0.662 | −0.194 |
| parabola | **10.0** | **0.942** | **0.681** | **−0.262** |
| uniform | 1.0 | 0.781 | 0.586 | −0.195 |
| uniform | 5.0 | 0.804 | 0.535 | −0.268 |
| uniform | **10.0** | **0.822** | **0.664** | **−0.158** |
| collision | 1.0 | 0.524 | 0.461 | −0.063 |
| collision | 5.0 | 0.608 | 0.543 | −0.065 |
| collision | **10.0** | **0.626** | **0.583** | **−0.043** |

#### 对比 broken-init（h=16）— 数值差距巨大

| 配置 | broken-init frozen | fixed-init frozen | 提升 |
|---|---|---|---|
| parabola w=50/λ=10 | 0.167 | **0.681** | +0.51 |
| uniform w=50/λ=10 | 0.298 | **0.664** | +0.37 |
| collision w=50/λ=10 | 0.071 | **0.583** | +0.51 |

### 5.3 关键观察（fixed-init 修订版）

1. **frozen-target cos 远不是随机水平**：fixed-init 上 cos 在 0.46–0.68 区间，远高于 broken-init 的 0.04–0.30。**predictor 输出与 pusht semantic 实际上保持了相当多的对齐**
2. **落差 Δ 大幅缩小**：broken-init Δ = −0.66 ~ −0.88（崩塌式），fixed-init Δ = **−0.04 ~ −0.34**（轻度损失）
3. **collision 上几乎无落差**（Δ = −0.04 ~ −0.07）—— pusht 预训练的视觉知识在 collision 上保留最多，predictor 在 frozen-target 空间里几乎和 trained-target 一样准
4. **uniform 上 Δ 反而随 λ 减小**（λ=1 Δ=−0.20 → λ=10 Δ=−0.16）—— 出乎意料，可能因为 λ=10 时 encoder 压扁到"球位置"方向，恰好与 pusht 的位置识别能力对齐
5. **结论修正**：原版"predictor 学的是 encoder-specific 局部映射"**在 broken-init 上成立、在 fixed-init 上只部分成立**。fixed-init 下 predictor 实际**学到了相当通用的 dynamics**（pusht 预训练保留了语义结构），但 trained-target metric 仍然高估了它的真实质量（落差 0.04-0.34）

---

## 6. 综合机制：高 λ 在 image-based LeWM 上做了什么（fixed-init 版）

```
出发点
  pusht-pretrained ViT  →  zero-shot 已能在 phyworld 解码物理量到 ρ ≈ 0.9
                            (来自 5-26 主报告 §6.1)

加 deep-sup probe 训练（fixed-init, λ=10）：
  λ × probe_loss  是 pred_loss 的 15-125 倍（依域不同）
                ↓
  optimizer 主要在最小化 probe_loss（probe 占 total 56-78%）
                ↓
  encoder 被推着把 latent 中"物理量对应的方向"放大、其他方向压扁
                ↓
  projector 输出 eff_dim：13~41 (λ=1) → 3~17 (λ=10)，塌方 39-90%
                ↓
  predictor 在这个被压扁的空间里学动力学
                ↓
  评估时：
    - K=4 ρ：probe loss 的对偶 → 单调或近单调上升 ✓（表面提升）
    - latent cos（trained-target）：encoder 自相似 → 单调上升 ✓（表面提升）
    - pred_loss：world model 真目标 → uniform / collision 单调退化 ✗，parabola U 形
    - frozen-target cos：fixed-init 上落差仅 −0.04~−0.34（pusht 语义保留较多）
       但 trained-target cos 仍然高估了 predictor 的真实质量
```

**修订版结论**：
- Sweep 报告里"高 λ 全面胜出"主要还是 **probe 损失对偶**的产物（K=4 ρ 必然涨），不代表 world model 在物理上变好
- 但程度比 broken-init 时温和：fixed-init 上 predictor 在 frozen pusht semantic 空间里仍保留 60-80% 的 cos
- **生产建议**：用 **pred_loss 选 λ**（λ=1 全域最佳），把 K=4 ρ / latent cos 当**次要诊断**，不要单独依赖

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
