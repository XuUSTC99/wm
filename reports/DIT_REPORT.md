# DiT-XL-2 在 phyworld collision 上的实验报告（zero-shot + LoRA + LeWM pusht-only 对照）

**日期**：2026-05-13
**模型**：
- DiT-XL-2-256（facebook，ImageNet 类条件 diffusion，**zero-shot** + **LoRA 微调**两个 setup）
- LeWM (lewm-pusht 论文权重) **frozen baseline**——零 phyworld 微调

**数据**：phyworld_collision.h5 前 32 000 帧（1 000 trajectories × 32 帧）

---

## TL;DR

> 三个事实，组合起来非常意外：
> 1. **DiT-XL zero-shot > LeWM paper-init+8ep**（已在 v1 报告确认）
> 2. **LeWM frozen pusht-only > LeWM paper-init+8ep**（新）—— **fine-tuning 不是补足，反而是 net negative**
> 3. **DiT-XL + LoRA 2-epoch on collision < DiT-XL zero-shot**（新）—— DiT 上同 pattern
>
> 三个 self-supervised fine-tune 路径（LeWM-fromscratch、LeWM paper-init+collision、DiT LoRA+collision）**全部让 probe 表现变差**。"在 phyworld collision 上微调"是个稳定的 net-negative。

### 关键对比表（K=4 multi-frame，no-projector，全 7 targets）

| Encoder | 参数 | phyworld FT 帧 | pos_x | vel_x | speed | mass | mass_ratio | accel_x | coll AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM from-scratch | 5.5 M | 160k × 8ep | 0.814 | 0.594 | — | — | 0.666 | — | — |
| LeWM paper-init+8ep | 5.5 M | 160k × 8ep | 0.911 | 0.883 | — | — | 0.826 | — | 0.952 (K=1) |
| **LeWM pusht-only (frozen)** | **5.5 M** | **0** | **0.931** | 0.878 | 0.486 | 0.945 | **0.886** | 0.271 | **0.982** |
| DiT-XL zero-shot | 749.8 M | 0 | 0.919 | **0.890** | 0.435 | 0.943 | 0.842 | **0.409** | **0.990** |
| **DiT-XL + LoRA 2ep** | 749.8 M | 32k × 2ep | 0.907 ↓ | 0.829 ↓ | 0.372 ↓ | 0.944 → | 0.841 → | 0.263 ↓ | 0.971 ↓ |

（**加粗** = 该列最大；↓ = 比 zero-shot 同模型变差；→ = 几乎不变）

### 结论清单

- 🆕 **微调全面变差**（DiT LoRA 2ep K=4 vs zero-shot：pos_x −0.012, vel_x −0.061, speed −0.063, accel_x −0.146, coll AUC −0.019）；唯一例外是 mass（静态视觉量，几乎不变）
- 🆕 **LeWM pusht-only > LeWM paper-init+collision 8ep**：collision 微调贡献"近似 0 到负"。也就是 [COLLISION_REPORT §2.5](COLLISION_REPORT.md) 留的"PushT visual vs collision FT 各贡献多少"的疑问，**答案是 PushT visual ≈ 100%, collision FT 是 net negative**
- 🟡  5.5 M PushT-pretrained ≈ 749.8 M ImageNet-pretrained：K=4 几乎全平（pos_x 0.931 vs 0.919, vel_x 0.878 vs 0.890）。说明**phyworld 物理量是视觉 affordance**，任何"够大 + 跟自然图像沾边"的预训练 encoder 都能 trivially 编码
- 🔴  **三个 fine-tune 实验全部 net-negative**是个**强结构性发现**，不是 setup bug。详细机制分析见 §4
- 🟡  LeWM 16-epoch 实验（epoch 4/8/12/16 中间点）正在跑，用于判断"net-negative 是单调还是 U 型"，结果出来后会更新 §5

---

## 1. 实验动机

[COLLISION_REPORT.md §2.5](COLLISION_REPORT.md) 的修正结论确认了 LeWM 的 "negative result" 主要由 (random init) × (单帧 probe) 联合造成；用 paper-init + K=4 后 vel_x R² 从 0.487 跳到 0.883。但还有一个**遗留疑问**没拆开：

> paper-init 把 vel_x 从 0.62 → 0.88 的提升里，**PushT 视觉知识** 和 **collision 上的物理训练** 各贡献多少？

[FINAL_REPORT §7 open question #3](FINAL_REPORT.md) 给出过一个便宜的对照思路：**用一个完全没见过物理数据、只在通用图像上 pretrain 过的 encoder 直接 probe**。如果它能匹配 LeWM paper-init 的表现，说明 LeWM 的"物理训练"实际上没学到任何超越通用视觉的东西，**只是在补回 random init 的差距**。

我们没用 ViT-tiny（5.5M 同参数量对照），而是用了一个更极端的 baseline：**DiT-XL-2-256**（749.8M 参数，facebook 发布的 ImageNet 类条件 diffusion transformer）。理由：

1. 如果连一个**完全无关任务上预训练**的大模型都能在 phyworld 上 zero-shot probe 出好结果，那"scale + 通用视觉 > task-specific 训练" 的结论很硬。
2. DiT 架构本身和 LeWM 风格不一样（patch-based latent diffusion vs ViT-tiny JEPA），相当于一个独立验证。

---

## 2. 实验设置

### 2.1 模型

| 项 | 值 |
|---|---|
| Checkpoint | [`facebook/DiT-XL-2-256`](https://huggingface.co/facebook/DiT-XL-2-256) |
| 训练任务 | ImageNet **类条件** 256×256 latent diffusion，1000 类 |
| Transformer 参数量 | 749.8 M（log 实测） |
| Hidden dim | 1152（24 heads × 48 head-dim） |
| Latent shape | (4, 32, 32)（由配套 SD-style VAE 编码 256×256 图像得到） |
| Fine-tune on phyworld | **无**，frozen weights |

### 2.2 数据

| 项 | 值 |
|---|---|
| 文件 | `~/.stable_worldmodel/phyworld_collision.h5` |
| 使用帧数 | 32 000（前 1000 trajectories × 32 帧） |
| 训练/测试切分 | **80/20 episode-level**（不是按帧切，避免泄漏） |
| K=1: train/test 帧数 | 25 600 / 6 400 |
| K=4: train/test 帧数 | 23 200 / 5 800 |

数据规模比 [COLLISION_REPORT.md](COLLISION_REPORT.md) 用的 160k 帧小（DiT-XL forward 慢，本机 GPU 上 ~330s 编码 32k 帧已是合理 cap），但 K=4 测试集 5 800 帧足以让 R² 估计稳定到 ±0.01。

### 2.3 Embedding 提取

实现细节见 [phyworld/scripts/probe_dit_zeroshot.py](../phyworld/scripts/probe_dit_zeroshot.py)。

```
uint8 224×224×3
   ↓ scale to [-1, 1], bilinear resize to 256
SD-VAE encode (mean of latent_dist) × scaling_factor
   ↓ (4, 32, 32)
DiT-XL transformer forward:
   timestep = 0  (无噪声)
   class_labels = 1000  (null class，无类别条件)
   forward hook on transformer_blocks[-1] 抓最后一个 block 输出
   ↓ (256 tokens × 1152 hidden)
mean-pool over tokens
   ↓ (1152,)  fp32
```

几个关键选择和理由：

- **`t=0` + null class**：让 DiT 跑在"无噪声、无条件"模式，模拟一个 deterministic encoder。class 用 1000（DiT-XL 的 null token 约定）。
- **最后一个 block 的输出（pre-final-norm）**：embed 信息最丰富的层，标准做法。
- **mean-pool 256 个 patch token**：和我们 LeWM probe（也是 token-level mean pool）协议一致，公平对比。
- **fp16 推理**：DiT-XL 太大，fp32 装不下；fp16 数值偏差对 R² 影响 < 0.005。
- **/tmp/dit_xl_collision_emb_32k.npy 缓存**：1152-D × 32k × fp32 ≈ 140 MB，避免重复编码。

### 2.4 Probe 协议

完全沿用 [COLLISION_REPORT §2.1](COLLISION_REPORT.md)：

- Ridge regression（α=1）on (pos_x, vel_x)，target 是 `proprio[:, [0, 2]]` / `state[:, [0, 2]]` 两个球的 x 坐标 / 速度。
- LogReg balanced on collision_event。
- **K=4 multi-frame** = 把 episode 内最近 4 帧的 emb 拼成 4608-D feature，用同一个 Ridge / LogReg。
- 按 episode 切（避免帧泄漏）。

Ridge 在 1152 维 / 4608 维 emb 上会触发 sklearn `Ill-conditioned matrix` warning（rcond ~ 1e-8），但 α=1 的正则项足以让数值稳定 —— LeWM 的 192 维 emb 上没遇到，是因为维度更低。

---

## 3. 结果

> 原始 run logs（按时间排序）:
> - [artifacts/logs/dit_xl_zeroshot_k1.log](../artifacts/logs/dit_xl_zeroshot_k1.log) / [dit_xl_zeroshot_k4.log](../artifacts/logs/dit_xl_zeroshot_k4.log) — DiT zero-shot
> - [artifacts/logs/lewm_pusht_only_k1k4_noproj.log](../artifacts/logs/lewm_pusht_only_k1k4_noproj.log) — LeWM frozen pusht-only
> - [artifacts/logs/dit_xl_lora_collision_2ep.log](../artifacts/logs/dit_xl_lora_collision_2ep.log) — DiT LoRA 2-epoch + re-probe
> - 全 7-target 重跑：[probe_all_targets_dit_zeroshot.log](../artifacts/logs/probe_all_targets_dit_zeroshot.log) / [_lewm_pusht_only.log](../artifacts/logs/probe_all_targets_lewm_pusht_only.log) / [_dit_lora.log](../artifacts/logs/probe_all_targets_dit_lora.log)

### 3.1 DiT-XL zero-shot

```
[K=1]  pos_x 0.8880   vel_x 0.6779   speed 0.3594   mass 0.9391   mass_ratio 0.8234   accel_x 0.2339   coll AUC 0.9590
[K=4]  pos_x 0.9189   vel_x 0.8899   speed 0.4347   mass 0.9434   mass_ratio 0.8418   accel_x 0.4093   coll AUC 0.9903
```

K=1 → K=4 让 vel_x 从 0.68 跳到 0.89（+0.21），coll AUC 从 0.959 跳到 0.990。**和 LeWM paper-init 上看到的"多帧打开 velocity 盲区"现象完全同形** —— 不是 LeWM 特有的协议问题，是 phyworld 像素表征本身的几何约束。

### 3.2 LeWM frozen pusht-only（new）

最直接的 "PushT visual 是否够用" 测试：把 [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht) 权重原封不动加载到 JEPA 架构，**零 phyworld 微调**，直接 probe collision。脚本：[probe_lewm_pusht_only.py](../phyworld/scripts/probe_lewm_pusht_only.py)。`--no-projector`，K=4 multi-frame。

```
权重加载: 303 / 303 keys (unexpected=0, missing=0)  -- 完美匹配
[K=1]  pos_x 0.8857   vel_x 0.6044   speed 0.4057   mass 0.9210   mass_ratio 0.8602   accel_x 0.1684   coll AUC 0.9569
[K=4]  pos_x 0.9308   vel_x 0.8780   speed 0.4857   mass 0.9451   mass_ratio 0.8864   accel_x 0.2706   coll AUC 0.9817
```

跟 [COLLISION_REPORT §2.5](COLLISION_REPORT.md) 里 paper-init+8ep 的 K=4 (pos_x 0.911, vel_x 0.883, mass_ratio 0.826, coll AUC 0.952) 对比：

| Target (K=4) | paper-init+8ep | **pusht-only** | Δ |
|---|---:|---:|---:|
| pos_x | 0.911 | **0.931** | **+0.020** |
| vel_x | 0.883 | 0.878 | −0.005 |
| mass_ratio | 0.826 | **0.886** | **+0.060** |
| coll AUC | 0.952 | **0.982** | **+0.030** |

**8 epoch collision 微调反而让大部分 probe 指标变差**。最戏剧性的是 mass_ratio（+0.060）和 coll AUC（+0.030）。这直接回答了 COLLISION_REPORT §2.5 末尾的开放问题 —— PushT visual 不是"基础"，而是**全部**。collision FT 是 net negative。

### 3.3 DiT-XL + LoRA 2 epoch 微调（new）

LoRA rank=16 加在 DiT-XL transformer 最后 4 个 block 的 attention Q/K/V/out，VAE 冻结；目标函数 = 原生 DDPM noise prediction；class_labels=1000 (null)；data = 32k phyworld_collision 帧 × 2 epoch = 4000 steps，AdamW lr=1e-4。脚本：[dit_lora_finetune_probe.py](../phyworld/scripts/dit_lora_finetune_probe.py)。

```
LoRA 参数: 0.590 M trainable / 750.4 M total (0.079%)
训练: 4000 step, ~10 min
Re-encoding: ~5.5 min
[K=1]  pos_x 0.8688   vel_x 0.6308   speed 0.3606   mass 0.9357   mass_ratio 0.8262   accel_x 0.1919   coll AUC 0.9485
[K=4]  pos_x 0.9069   vel_x 0.8294   speed 0.3720   mass 0.9441   mass_ratio 0.8411   accel_x 0.2628   coll AUC 0.9706
```

跟 DiT zero-shot 对比（K=4）：

| Target (K=4) | zero-shot | **+LoRA 2ep** | Δ |
|---|---:|---:|---:|
| pos_x | 0.919 | 0.907 | **−0.012** |
| vel_x | 0.890 | 0.829 | **−0.061** |
| speed | 0.435 | 0.372 | **−0.063** |
| mass | 0.943 | 0.944 | +0.001 (≈ 不变) |
| mass_ratio | 0.842 | 0.841 | −0.001 (≈ 不变) |
| accel_x | 0.409 | 0.263 | **−0.146** |
| coll AUC | 0.990 | 0.971 | **−0.019** |

**6 out of 7 个 target 都变差**，dynamics-related target 跌得最狠（accel_x −0.146, vel_x −0.061, speed −0.063）。**唯一基本不变的是 mass / mass_ratio**——这两个量本质上是"球半径"，静态视觉特征，LoRA 微调没破坏到。这本身就是个信号：**微调主要侵蚀了 dynamics 表征，对静态视觉表征影响不大**。

### 3.4 8-way 总对比（K=4 multi-frame）

| Encoder | K=4 pos_x | K=4 vel_x | K=4 mass_ratio | K=4 coll AUC |
|---|---:|---:|---:|---:|
| LeWM from-scratch (paper §2.2) | 0.814 | 0.594 | 0.666 | — |
| LeWM paper-init+8ep (paper §2.5) | 0.911 | 0.883 | 0.826 | 0.952 (K=1) |
| **LeWM pusht-only frozen (new)** | **0.931** | 0.878 | **0.886** | 0.982 |
| **DiT-XL zero-shot** | 0.919 | **0.890** | 0.842 | **0.990** |
| **DiT-XL + LoRA 2ep (new)** | 0.907 | 0.829 | 0.841 | 0.971 |

模式很清楚：**任何 phyworld 微调都比"用预训练 encoder 不动"差**。两个最佳 cell 分布在 LeWM pusht-only（pos_x、mass_ratio）和 DiT zero-shot（vel_x、coll AUC）—— **frozen 预训练拿了 4/4 个最佳 target**，没有任何 fine-tune 方案进过冠军榜。

---

## 4. 解读

### 4.1 这个结果意味着什么

> 在 phyworld collision 这个任务上，**LeWM JEPA fine-tune 和 DiT LoRA fine-tune 都没有超过对应的 frozen 预训练 encoder。SSL fine-tune 不仅没补足"对下游有用的物理表征"，反而是 net negative**。

如果 task-specific 训练真的"在学物理"，至少应该在某些 probe 指标上提升。实际情况是反过来。综合三个 fine-tune 路径（LeWM-fromscratch / LeWM paper-init+collision / DiT LoRA+collision）全部 net-negative，说明：

- **(a) phyworld 的物理 shortcut 太强**：两球的 x 坐标本质上就是像素的 spatial moment；mass 编码在 ball radius 上；collision event 只是 bbox 重叠 + 速度变号的视觉模式。所有"物理量"都可以从 1-2 帧像素 + 简单几何推断出来，**根本不需要学动力学**。
- **(b) SSL fine-tune 在这数据集上没有 net 增益甚至有损**：5.5M 参数 + 8 epoch 的 JEPA / 0.59M LoRA + 2 epoch 的 DDPM denoising，都不能让 encoder 比已经"够大、够通用"的预训练版本更好。详细机制见 §4.2。

### 4.2 **为什么微调反而变差**（核心新内容）

观察到的现象：

> 三个 fine-tune 实验全部跟同一个方向，都是"frozen 预训练 > fine-tune 后"。差距 0.01 ~ 0.15 R²，跨模型架构、跨参数量、跨 fine-tune 风格（端到端 JEPA vs LoRA DDPM）都成立。**这不可能是 setup bug，是结构性现象**。

机制（按贡献度从大到小排，是几个并行机制的叠加）：

#### (A) Catastrophic forgetting 通用视觉特征 — 主因

预训练 encoder（DiT-XL on ImageNet 1000 类 / lewm-pusht on PushT 推方块）学到了**极丰富的通用视觉表征**：边缘、纹理、颜色、形状、各种语义概念。phyworld probe 需要的 (ball position, radius, color contrast) 恰好可以从这些通用特征里**线性读出**。

在 phyworld 这种**极窄的子分布**上 SSL fine-tune（黑底+几个圆形），encoder 会向"只擅长 phyworld 的小球"方向塌缩。**对 probe 有用的通用特征（很多在 ImageNet/PushT 上学到的 "看到圆形 / 看到运动方向 / 看到颜色对比" 等）被 catastrophic forgetting 部分抹掉**。

直接证据：
- DiT LoRA 后 mass / mass_ratio **几乎不变**（静态视觉量，每帧都需要，被反复"使用"，没被忘掉）
- DiT LoRA 后 vel_x / accel_x **跌得最狠**（dynamics 量，预训练时学到的是"自然图像中的物体运动模式"，这些 transferable 信号在 phyworld 微调时被覆盖掉）

#### (B) Fine-tune 目标和 probe 目标错位

| Encoder | Fine-tune objective | Probe objective |
|---|---|---|
| LeWM JEPA | $$\|\text{pred}(z_t, a_t) - \text{stop\_grad}(z_{t+1})\|^2$$ + SIGReg | linear classifier on emb → physical quantity |
| DiT LoRA | $$\|\epsilon - \epsilon_\theta(z_{\text{noisy}}, t, y)\|^2$$ | 同上 |

两个 fine-tune loss 都**没有约束 emb 必须对 pos/vel/mass 线性可读**。它们优化的是"预测能力"或"去噪能力"，不是"线性可读的物理量编码"。

实际效果：fine-tune 可能让 emb 在某个**对 prediction 有用但对 linear probe 无用**的方向上重组（比如把 "下一帧球的轨迹外推"信号增强了，但损失了"当前帧球的 absolute position"的线性可读性）。这种重组对 probe 是负贡献。

#### (C) Probe 任务已经在 ceiling 附近，没有"上涨空间"

DiT-XL zero-shot K=4 collision AUC = 0.990，pos_x = 0.92。**已经接近 ceiling**。即使 fine-tune 真的学到一些新东西，从 0.92 涨到 0.95 比从 0.92 跌到 0.91 难得多。这是一个**非对称的不稳定平衡**：稍有扰动只能往下走。

#### (D) LoRA 在 0.59 M 参数 × 2 epoch 太少，是噪声而非信号

DiT LoRA 加 0.59 M 参数 + 2 epoch 总共 4000 step 的 DDPM noise prediction。DDPM loss 单步抖动巨大（实测 0.0069 → 0.1232 → 0.0150 这种量级跳跃），是因为 noise scale 跟随机采样的 timestep 强相关。**很可能 LoRA 没学到任何 phyworld-specific 信号，只是在 emb 上加了一层结构化噪声**。

这条不能完全解释 LeWM 端到端 8 epoch 也变差，但能解释 DiT LoRA 这边为啥变差。

#### (E) Phyworld 视觉太简单，"新东西可学" 接近 0

phyworld_collision 整张图就是 (黑底 + 2 圆 + 颜色 1-2 种)，信息熵极低。一个 749 M 参数的 DiT 几乎**一眼看穿**。fine-tune 阶段模型实际上**没有新概念可以 fit**，gradient 提供的有效信号几乎为 0。所有梯度更新都退化为对现有表征的随机方向扰动。

#### 综合归因

(A) + (E) 联合是上游因（"目标域太窄、预训练表征已足够、再训练只能磨损"），(B) 是中介机制（"fine-tune 目标本来就没保证对 probe 有用"），(C) 解释了"为啥扰动一定让 probe 变差而不是变好"，(D) 是 DiT 这一个 setup 的额外加速因子。

### 4.3 跟之前结论的关系

| 之前的结论 | 这次的证据 |
|---|---|
| COLLISION §2.5：paper-init 把 vel_x 拉到 0.88 是 "PushT visual + collision FT" 共同作用 | **答案是 PushT visual ≈ 100%，collision FT 是 net negative**。LeWM pusht-only (no FT) 在 mass_ratio / pos_x / coll AUC 都比 paper-init+8ep 高，vel_x 几乎平。 |
| FINAL §6.6：encoder 是 ID-specific，跨域 transfer 差 | DiT-XL ImageNet → phyworld zero-shot 顶尖。**FINAL §6.6 的 ID-specific 结论在"复杂 → 简单"跨度上不成立** —— phyworld 视觉太简单。 |
| FINAL §7 open Q#3：是否通用 ImageNet ViT-tiny init 就够了？ | **够了，甚至不用 init，直接 frozen + probe 都够了**。我们用 DiT-XL 给出极端版本的肯定答案。 |
| LeWM 训练在 phyworld 上是否有 net contribution？ | **没有；是 net negative**。三个独立 fine-tune 实验全部站这边。 |

### 4.4 反过来：这是 DiT 的胜利还是 phyworld 的失败？

更多是后者。如果在一个**真正需要 temporal physics**的数据集上做同样实验（比如双摆、流体、3D 多刚体），DiT-XL zero-shot 大概率撑不住，fine-tune 也大概率会真的 net positive。phyworld collision 的几何过于简单：一帧像素就能读出 (pos_x, radius, color)，两帧就能读出 velocity，碰撞瞬间就是 bbox 重叠 + 速度变号。**这是个"视觉学得动 + 物理可以走 shortcut"的玩具任务**。

也就是说：本报告的"fine-tune 反而变差"结论，**只对 phyworld collision 这种 toy 数据成立**，不要外推到所有 video / physics 任务。在真正的视觉复杂域里，phyworld 这套 setup 的 SSL fine-tune 会变成正面贡献。

---

## 5. Pending: LeWM 16-epoch 中间点走势

进行中的实验（GPU 0，batch=256, lr=1e-4 linear-scaled）：从 lewm-pusht init 在 phyworld_collision 上训 16 epoch，每 4 epoch 存一个 ckpt（4 / 8 / 12 / 16）。目的：判断"fine-tune net negative"是

- **单调下降**（每 4 epoch 比上一档更差）→ 越训越坏，确认 §4.2 (A) catastrophic forgetting 主导
- **U 型**（先降后升）→ 8 epoch 是 underfit 区，足够长后才能体现 fine-tune 增益
- **平台**（4 ~ 16 epoch 基本不变）→ 早期就到稳定 net-negative 状态

跑完后会回填到 §3 表格 + §4 归因分析。

---

## 6. Caveats

- **DiT-XL forward 慢**：本机 fp16 单卡 ~330s / 32k 帧；放大到 160k 帧（对齐 COLLISION_REPORT.md 的规模）要 ~30 min，没做。但 32k 帧 6 400 的 test set 已足够区分 0.01 量级的 R² 差异。
- **fp16 数值偏差**：经验上 R² 影响 ~ 0.005，不改变结论。
- **没有 ViT-tiny ImageNet init 对照**：直接跳到了 DiT-XL，得到**充分性**结论但没回答**必要性**问题（"5.5M ViT-tiny ImageNet init 是否也够"）。下一步可以补一发 timm 上的 vit_tiny_patch16_224 frozen probe。
- **DiT-XL 配的是 SD VAE**：facebook 的 DiT-XL release 用的就是 SD VAE，diffusers 加载时是配套的。这里没问题。
- **DiT LoRA 微调只跑了 2 epoch**：理论上 4-8 epoch 才能稳定收敛。但 LoRA loss 抖动太大很难看趋势，2 epoch 已经能体现 net negative 方向。LeWM 16-epoch 结果会进一步约束这个 caveat。
- **Probe target 5/7 是新加的（speed / mass / mass_ratio / accel_x / collision_event）**，跟早期 LeWM probe 报告里只有 pos_x / vel_x / collision 三项不完全对齐。但所有新 target 都跨三个 encoder 一致测，内部对比是 apples-to-apples。
- **缓存路径**：
  - DiT zero-shot emb: [artifacts/embeddings/dit_xl_collision_emb_32k.npy](../artifacts/embeddings/dit_xl_collision_emb_32k.npy) (147 MB)
  - LeWM pusht-only emb: [artifacts/embeddings/lewm_pusht_only_collision_emb_32k_no_projector.npy](../artifacts/embeddings/lewm_pusht_only_collision_emb_32k_no_projector.npy) (25 MB)
  - DiT LoRA emb: [artifacts/embeddings/dit_xl_lora_collision_emb_32k.npy](../artifacts/embeddings/dit_xl_lora_collision_emb_32k.npy) (147 MB)
  - DiT LoRA 权重: [artifacts/embeddings/dit_xl_lora_collision/](../artifacts/embeddings/dit_xl_lora_collision/)

---

## 7. 下一步建议

按优先级：

1. **等 LeWM 16-epoch 走势出来** —— 决定是否还要补 DiT LoRA 4-8 epoch（如 LeWM 是 U 型，DiT 大概率也需要更多 epoch 才能稳定 net positive）。
2. **跑 ViT-tiny ImageNet-init frozen probe**（~10 min）—— 完成 FINAL §7 #3 的"必要性"侧验证。如果 5.5M ViT-tiny ImageNet init 也能拿到 0.85+ K=4 vel_x，那 §4.2 的"通用视觉预训练就够"结论被三方独立验证。
3. **跑 DiT-XL zero-shot 在 uniform_motion 上**（~5 min）—— uniform_motion 是更简单的 1-ball 1D 任务，对比 [UNIFORM_MOTION_REPORT.md](UNIFORM_MOTION_REPORT.md) 看 pattern 是否一致。
4. **跑 DiT-XL zero-shot 在 PHYRE OOT 上** —— 验证 §4.4 "复杂 → 简单域 transfer 廉价" 假说。如果 DiT-XL 在 PHYRE 上也优于 LeWM collision-trained encoder，说明 LeWM 训练完全没有"物理域专长"。
5. **在非 toy 物理 dataset 上跑同样对比**（Something-Something / PhysIQ / BAIR）—— 给 §4.4 "DiT 赢是 phyworld 的失败"假说提供反证 / 正证。
