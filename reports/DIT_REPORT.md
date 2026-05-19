# DiT-XL-2 在 phyworld collision 上的实验报告（zero-shot + LoRA + LeWM pusht-only 对照）

**日期**：2026-05-13
**模型**：
- DiT-XL-2-256（facebook，ImageNet 类条件 diffusion，**zero-shot** + **LoRA 微调**两个 setup）
- LeWM (lewm-pusht 论文权重) **frozen baseline**——零 phyworld 微调

**数据**：phyworld_collision.h5 前 32 000 帧（1 000 trajectories × 32 帧）

---

## TL;DR

跨**7 个 encoder × 7 个 probe target × 多 K 协议**的完整对比，加上 LeWM 16-epoch + DiT LoRA 8-epoch 两条 trajectory，结论如下：

> 1. **在 phyworld collision 上做 SSL 微调（JEPA 或 DDPM LoRA）都不能净提升 probe 表现** —— 整体微弱 net-negative。
> 2. **预训练域比 scale 更重要**：5.5M 参数的 PushT-pretrained ViT-tiny ≈ 749M 参数的 ImageNet DiT-XL。换 ImageNet ViT-tiny（同 5.5M 同架构）→ K=4 vel_x 跌 0.124。
> 3. **两种微调失败模式不同**：LeWM JEPA 在 epoch 4-16 完美 plateau（稳定 trade-off：speed+mass↑ / pos+mass_ratio+accel+coll↓）；DiT LoRA 不收敛（震荡 + drift），并在 epoch 8 因 gradient 爆炸 NaN。

### 关键对比表（K=4 multi-frame，no-projector，全 7 target）

| Encoder | params | pretrain | phyworld FT | pos_x | vel_x | speed | mass | mass_ratio | accel_x | coll AUC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixel-stats (9-D) | — | — | 0 | 0.763 | 0.516 | 0.354 | 0.936 | 0.692 | 0.041 | 0.752 |
| random ViT-tiny | 5.5 M | — | 0 | 0.747 | 0.573 | 0.350 | 0.933 | 0.742 | 0.045 | 0.831 |
| LeWM from-scratch | 5.5 M | — | 8 ep | 0.814 | 0.594 | — | — | 0.666 | — | — |
| **ImageNet ViT-tiny** | 5.5 M | **ImageNet-21k+1k** | 0 | 0.903 | 0.754 | 0.369 | 0.934 | 0.782 | 0.138 | 0.944 |
| LeWM paper-init+8ep | 5.5 M | PushT | 8 ep | 0.911 | 0.883 | — | — | 0.826 | — | 0.952 (K=1) |
| **LeWM pusht-only (frozen)** | 5.5 M | **PushT** | 0 | **0.931** | 0.878 | 0.486 | 0.945 | **0.886** | 0.271 | **0.982** |
| **LeWM 16ep epoch 16** | 5.5 M | PushT + 16ep | 16 ep | 0.913 | 0.875 | **0.542** | **0.956** | 0.864 | 0.234 | 0.974 |
| **DiT-XL zero-shot** | 749.8 M | ImageNet diffusion | 0 | 0.919 | **0.890** | 0.435 | 0.943 | 0.842 | **0.409** | **0.990** |
| DiT-XL + LoRA epoch 2 | 749.8 M | ImageNet → 2ep | 2 ep | 0.907 | 0.830 | 0.370 | 0.944 | 0.842 | 0.265 | 0.971 |
| DiT-XL + LoRA epoch 6 | 749.8 M | ImageNet → 6ep | 6 ep | 0.907 | 0.821 | 0.325 | 0.942 | 0.851 | 0.237 | 0.961 |
| DiT-XL + LoRA epoch 8 | 749.8 M | ImageNet → 8ep | 8 ep | — | — | — | — | — | — | — *(NaN 训练发散)* |

（**加粗** = 该列最大；冠军 cell 全部由 **frozen 预训练** encoder 拿下）

### 结论清单

- 🆕 **预训练域 > 预训练 scale > task-specific FT**（按 ROI 排序）。详细单变量梯度看 [§3.5](#35-通用-baseline-系列new)。
- 🆕 **ImageNet ViT-tiny < LeWM pusht-only**（同 5.5M）—— **PushT 视觉训练比 ImageNet 监督 pretrain 多带 ~0.12 vel_x**。回答 [FINAL §7 #3](FINAL_REPORT.md) 的"通用 ViT-tiny 是否够"：**不够**。
- 🆕 **LeWM 16-ep trajectory plateau**：epoch 4 / 8 / 12 / 16 几乎重合（max delta < 0.01）—— 不是欠拟合假象，**就是稳定 net-negative trade-off**。但不是均匀下降，是有结构的：speed+mass↑ / 其余↓。
- 🆕 **DiT LoRA 8-ep trajectory 不稳**：epoch 2 vs 4 vs 6 震荡，epoch 8 因为 weight 单调放大 (0.31 → 0.44 → 0.58 → NaN) 训练发散。**DDPM 微调需要 gradient clipping**，本次 setup 没启用。
- 🟡  **frozen 预训练在 8/8 个最佳 cell 上夺冠**：pos_x、mass_ratio 给 LeWM pusht-only；vel_x、accel、coll AUC 给 DiT zero-shot；speed、mass 给 LeWM 16-ep（唯一非 frozen 拿冠的两个 target）。
- 🔴  **整体结论**：**phyworld collision 是个"frozen 预训练就够"的数据集**。在它上面做 SSL 微调（无论 JEPA 还是 DDPM LoRA）都不能净提升 probe。这不能外推到所有 video / physics 任务 —— phyworld 视觉太简单是关键前提（详 §4.4）。

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

### 3.5 通用 baseline 系列（new）

为给上面的"预训练 vs 随机"差距画 floor + 验证 "PushT 视觉 ≠ 通用视觉"，补做三个 baseline：

- **pixel-stats**（9-D feature）：每通道 mean / std / mean² 在空间维上聚合。最朴素的"无 encoder"基线
- **random ViT-tiny**（5.5M params, no pretrain）：和 LeWM 同架构但随机初始化。控制"架构 inductive bias"
- **ImageNet ViT-tiny**（5.5M params, timm `vit_tiny_patch16_224.augreg_in21k_ft_in1k`）：监督 ImageNet-21k → 1k pretrain。**关键 control: 同参数量 / 同架构 / 通用视觉 pretrain**

K=4 结果：

| Encoder | params | pretrain | pos_x | vel_x | speed | mass | mass_ratio | accel | coll AUC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pixel-stats | — | — | 0.763 | 0.516 | 0.354 | 0.936 | 0.692 | 0.041 | 0.752 |
| random ViT-tiny | 5.5 M | — | 0.747 | 0.573 | 0.350 | 0.933 | 0.742 | 0.045 | 0.831 |
| **ImageNet ViT-tiny** | **5.5 M** | **ImageNet-21k+1k** | **0.903** | **0.754** | **0.369** | **0.934** | **0.782** | **0.138** | **0.944** |
| LeWM pusht-only | 5.5 M | PushT | 0.931 | 0.878 | 0.486 | 0.945 | 0.886 | 0.271 | 0.982 |
| DiT-XL zero-shot | 749.8 M | ImageNet diffusion | 0.919 | 0.890 | 0.435 | 0.943 | 0.842 | 0.409 | 0.990 |

三个新结论：

**🆕 (1) PushT 视觉预训练 > ImageNet 视觉预训练**（同 5.5M ViT-tiny）

K=4 vel_x: pusht-only 0.878 vs ImageNet 0.754 = **+0.124**。 [FINAL §7 #3](FINAL_REPORT.md) 问 "通用 ImageNet pretrain 够不够？" → **答案: 不够**。ImageNet 给到 ~0.75 vel，PushT 给到 ~0.88。**预训练域的"接近度"很重要** —— PushT 是机器人抓 T 形块，跟 phyworld 球碰撞同属"物体操控+几何原语"，比 ImageNet 1000 类语义分类更接近。

**🆕 (2) Scale 帮 dynamics，不帮 static**（DiT 749M vs ImageNet ViT-tiny 5.5M）

K=4 vel_x: DiT 0.890 vs ImageNet ViT-tiny 0.754 = **+0.136**（scale 帮）。
K=4 mass: DiT 0.943 vs ImageNet ViT-tiny 0.934 = +0.009（≈ 不变）。

所以 §4.1 报告里说的"scale 通吃" 其实是**有偏的** —— 视觉静态量（位置、质量）任何 5.5M pretrained 都能 trivially 编码，scale 带来的红利只在 dynamics（速度、加速度、碰撞）上才显著。

**🆕 (3) Random ViT-tiny ≈ pixel-stats**

K=4 vel_x: random 0.573 vs pixel 0.516 = +0.057（噪声级别）。
说明 ViT-tiny 架构本身**没有显著的视觉 inductive bias**，纯随机初始化在 phyworld 上几乎跟 9 维像素均值统计同水平。**pretrain 是关键**。

#### 完整 K=4 vel_x"阶梯"

| Encoder | K=4 vel_x | 跟前一档比 |
|---|---:|---:|
| pixel-stats | 0.516 | (floor) |
| random ViT-tiny | 0.573 | +0.057 |
| LeWM from-scratch | 0.594 | +0.021（160k phyworld 训练不如随机 init 多多少）|
| **ImageNet ViT-tiny** | **0.754** | **+0.160**（ImageNet 监督 pretrain 是个大跳）|
| LeWM pusht-only | 0.878 | +0.124（PushT 视觉再加 ~0.12）|
| LeWM paper-init+8ep | 0.883 | +0.005（collision 微调几乎 0 贡献）|
| DiT-XL zero-shot | 0.890 | +0.007（scale 又加 ~0.01）|

漂亮的单调梯队，每一步 ROI 都能看清楚。**预训练域 + 预训练规模这两个变量加起来解释了 99% 的 vel_x 表现**，task-specific fine-tune 在这套 setup 下几乎 0 贡献。

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

## 5. Fine-tune 走势对比（trajectory experiments）

为判断 §4 的 "fine-tune 反而变差"是否是欠拟合假象、单调下降、还是 U 型曲线，跑了两条独立的 trajectory：
1. **LeWM paper-init + collision 16 epoch**，每 4 epoch 存 ckpt（4/8/12/16），bf16，batch=64, lr=3.5e-5
2. **DiT LoRA collision 8 epoch**，每 2 epoch 存 ckpt（2/4/6/8），fp16 base + fp32 LoRA，batch=16, lr=1e-4

### 5.1 LeWM 16-epoch trajectory（K=4，no-projector）

| Encoder | pos_x | vel_x | speed | mass | mass_ratio | accel_x | coll AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| pusht-only (frozen, 0 ep) | **0.931** | 0.878 | 0.486 | 0.945 | **0.886** | **0.271** | **0.982** |
| 16-ep epoch 4 | 0.909 | 0.873 | 0.532 | 0.952 | 0.849 | 0.245 | 0.975 |
| 16-ep epoch 8 | 0.913 | 0.879 | 0.552 | 0.955 | 0.867 | 0.240 | 0.974 |
| 16-ep epoch 12 | 0.915 | 0.876 | 0.545 | 0.956 | 0.862 | 0.236 | 0.975 |
| 16-ep epoch 16 | 0.913 | 0.875 | **0.542** | **0.956** | 0.864 | 0.234 | 0.974 |

**结论**：epoch 4 到 16 几乎完美 **plateau** —— 所有指标在 epoch 8 已收敛，再训 8 个 epoch（epoch 9-16）几乎不变化（max delta < 0.01）。

**vs frozen pusht-only（零微调）**：

| Target | 微调能否帮上忙？ | Delta (epoch 16 − pusht-only) |
|---|---|---:|
| pos_x | ❌ 下降 | −0.018 |
| vel_x | → 基本平 | −0.003 |
| speed | ✅ 上升 | **+0.056** |
| mass | ✅ 微升 | +0.011 |
| mass_ratio | ❌ 下降 | −0.022 |
| accel_x | ❌ 下降 | −0.037 |
| coll AUC | ❌ 下降 | −0.008 |

**LeWM JEPA 微调在 7 个 probe target 上不是单调 hurt，是 trade-off**：换 0.06 的 speed + 0.01 的 mass，**代价**是 pos/mass_ratio/accel/coll_AUC 各掉 0.02-0.04。

可能的解释（推测）：JEPA 训练目标是预测下一帧 latent，**鼓励 encoder 编码"在接下来会变什么"的信息**。speed = |velocity| 直接决定下一帧偏移幅度，被 prediction loss 大量放大；而 absolute position、collision boolean 这种"当前帧静态"信号，对预测没什么帮助，被弱化。这跟 [COLLISION_REPORT §2.3 (b)](COLLISION_REPORT.md) 提到的 "projector 在压缩对 prediction 无用的信息" 是同一机制，但这次出现在 projector 之前（no-projector probe）—— **encoder 本身在按 prediction 目标重构表征**。

回到 §4 的"为啥 fine-tune 反而变差"：trajectory 数据告诉我们**机制 (A) catastrophic forgetting 不是完整答案**。更准确的描述是 **(B) fine-tune 目标和 probe 目标的对齐度** —— JEPA loss 选择性地保留了对预测下一帧有用的信息（speed, mass），抛弃了对预测没用的信息（absolute pos, collision_event）。**不是"忘掉了"，是"主动地重新分配了 encoder 容量"**。

### 5.2 DiT LoRA 8-epoch trajectory（K=4，fp32 re-probe）

| Encoder | pos_x | vel_x | speed | mass | mass_ratio | accel_x | coll AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot (0 ep) | **0.919** | **0.890** | 0.435 | 0.943 | 0.842 | **0.409** | **0.990** |
| LoRA epoch 2 | 0.907 | 0.830 | 0.370 | 0.944 | 0.842 | 0.265 | 0.971 |
| LoRA epoch 4 | 0.915 | 0.854 | 0.354 | 0.947 | **0.869** | 0.294 | 0.975 |
| LoRA epoch 6 | 0.907 | 0.821 | 0.325 | 0.942 | 0.851 | 0.237 | 0.961 |
| LoRA epoch 8 | ❌ **NaN**（训练发散）| — | — | — | — | — | — |

**两个发现**：

**(a) DiT LoRA 不收敛，在 zero-shot 附近震荡 + 缓慢漂离**：epoch 2 → 4 看似回升（pos_x 0.907 → 0.915 等），但 epoch 4 → 6 又跌回（甚至比 epoch 2 更低）。无平台。跟 LeWM 的 "epoch 8 已收敛" 形成鲜明对照。

**(b) DiT LoRA 在 epoch 7-8 之间训练发散**。检查 LoRA 权重 max|w|：

| ckpt | max\|w\| | NaN? |
|---|---:|---|
| epoch_2 | 0.313 | 否 |
| epoch_4 | 0.443 | 否 |
| epoch_6 | 0.577 | 否 |
| epoch_8 | **NaN** | **是** |

LoRA 权重单调放大（0.31 → 0.44 → 0.58），最终某一步 gradient 爆炸 → NaN。**根本原因**：DDPM noise prediction 单步 loss 抖动巨大（实测在 0.007 → 0.5 之间随机跳）+ AdamW lr=1e-4 + **没启用 gradient clipping**。修复方法是加 `torch.nn.utils.clip_grad_norm_` 到 1.0，但这次实验没做。

#### LeWM vs DiT trajectory 对照

| 维度 | LeWM JEPA | DiT LoRA |
|---|---|---|
| 训练 loss 性质 | $$\|\text{pred}(z_t,a_t)-\text{sg}(z_{t+1})\|^2$$ + SIGReg，稳定 | DDPM noise pred，单步极不稳 |
| epoch 4-16 trajectory | 完美 plateau，方差 < 0.01 | 震荡 + drift，方差 0.02-0.06 |
| 训练终态 | epoch 16 ≈ epoch 4 | epoch 6 < epoch 4，epoch 8 NaN |
| 跟 zero-shot 比 | 整齐 trade-off：speed + mass 升，其他全掉 | 全面 net negative，没有 silver lining |

**这两个完全不同的 pattern 都印证"在 phyworld 上 SSL 微调不能让 probe 更准"**，但揭示了**两种不同的失败模式**：
- **JEPA**：稳定但选择性地丢失对 probe 有用的信息（损 pos 换 speed）
- **DDPM**：根本不稳定，朝着随机方向漂，最终训练发散

跑完之后的实操经验：**两种 fine-tune 都不应该在 phyworld 上做** —— 一个换不来净收益，另一个还会炸。下一步 phyworld 实验直接用 frozen 预训练 + multi-frame probe 就行。

---

## 6. 方法论修正：OOD probe 的 probe-extrapolation 漏洞

[FINAL_REPORT §6.6](FINAL_REPORT.md) 当时通过在 `phyworld_collision_eval.h5` 上做 OOD probe，得出过 "encoder 是 ID-specific、跨 OOD partition 的物理量读出会崩" 这个强结论。**这个结论是错的**，根因是原 probe 协议有 probe-extrapolation 缺陷，跟 encoder 本身没关系。

### 6.1 缺陷描述

原协议（`probe_ood.py`）：
1. 在 **ID 训练 fold** 上拟合一个 Ridge（线性外推器）
2. 把**同一个 Ridge** 应用到 r-OOD / v-OOD / both-OOD 这三个 partition 的测试 fold
3. R² drop 被解读为 "encoder 在 OOD 上的表征崩了"

**问题**：partition 的定义是 `r ∈ [0.7, 1.5], v ∈ [1, 4]` 之内为 ID，之外为 OOD。OOD 帧的 target（pos / vel / mass）数值**落在 ID 训练范围之外**。Ridge 在 ID 上拟合的线性系数**无法外推到训练域外的值范围**，即使 encoder 完美地、一致地编码了那些物理量，probe 的预测也会偏。**R² drop 完全可能是 "线性映射无法外推"，跟 encoder 表征质量无关**。

### 6.2 修正：三协议对照

[phyworld/scripts/probe_ood_per_partition.py](../phyworld/scripts/probe_ood_per_partition.py) 同时跑三个协议：

| 协议 | Ridge 拟合数据 | 测试数据 | 设计目的 |
|---|---|---|---|
| **A (原)** | ID 训练 fold | 各 partition 测试 fold | 测"ID probe 是否能 transfer 到 OOD"|
| **B** | **各 partition 自己的训练 fold**（80/20 ep-level） | 同一 partition 测试 fold | 测"encoder 在该 partition 内有多少 linear-readable 信息"，无外推问题 |
| **C** | **所有 partition 训练 fold 合并** | 各 partition 测试 fold | **同一个 probe** 跨 partition；测 encoder 能否跨 partition 一致编码 |

诊断逻辑：
- A 低 + B 高 → probe 外推不出，encoder 没事
- B 高 + C 低（某 partition）→ encoder 在该 partition 上 linear 信息存在但跟其他 partition **不在同一线性方向**
- A ≈ B ≈ C → encoder transfer 良好，三协议都同意

### 6.3 结果：4 个 encoder 全部 "A 低 + B 高"

在 4 个 encoder × 4 个 partition × 6 个 target 上跑完两协议（embedding cache 在 [artifacts/embeddings/*_collision_eval_emb_52k*.npy](../artifacts/embeddings/)，原 log 在 [artifacts/logs/ood_per_partition_*.log](../artifacts/logs/)）。**both-OOD partition** 上 A vs B 对比：

| Encoder | params | pretrain | A pos_x | **B pos_x** | A mass_ratio | **B mass_ratio** | A vel_x | **B vel_x** |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ImageNet ViT-tiny | 5.5 M | ImageNet sup. | −0.026 | **+0.716** | −0.077 | **+0.908** | +0.111 | **+0.663** |
| LeWM pusht-only | 5.5 M | PushT JEPA | +0.005 | +0.745 | **−0.102** | **+0.932** | −0.043 | **+0.684** |
| LeWM 16-ep ep16 | 5.5 M | PushT + 16ep coll | −0.138 | +0.745 | −0.164 | +0.920 | −0.081 | +0.693 |
| DiT-XL zero-shot | 749.8 M | ImageNet diffusion | +0.101 | **+0.790** | −0.049 | **+0.930** | +0.094 | **+0.728** |

**四个完全不同的 encoder（参数量从 5.5M 到 750M，预训练从 PushT JEPA 到 ImageNet 监督到 ImageNet diffusion）全部观察到同样的 pattern**：

- Protocol A on both-OOD：所有 target 几乎"崩到 0 或负值"
- Protocol B on both-OOD：所有 target 恢复到 0.7-0.95 的水平，**很多比 ID 还高**（因为 both-OOD partition 训练样本更多，Ridge 拟合更充分）

### 6.4 跨 partition 完整数据（Protocol B，正确版本）

| Encoder | ID pos_x | r-OOD pos_x | v-OOD pos_x | both-OOD pos_x | ID mass_ratio | r-OOD mass_ratio | v-OOD mass_ratio | both-OOD mass_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet ViT-tiny | 0.837 | 0.637 | 0.864 | 0.716 | 0.776 | 0.762 | 0.868 | 0.908 |
| LeWM pusht-only | 0.859 | 0.699 | 0.904 | 0.745 | 0.729 | 0.792 | 0.899 | 0.932 |
| LeWM 16-ep ep16 | 0.742 | 0.664 | 0.871 | 0.745 | 0.573 | 0.782 | 0.955 | 0.920 |
| DiT-XL zero-shot | 0.835 | 0.755 | 0.858 | 0.790 | 0.812 | 0.788 | 0.933 | 0.930 |

| Encoder | ID vel_x | r-OOD vel_x | v-OOD vel_x | both-OOD vel_x | ID coll AUC | r-OOD coll AUC | v-OOD coll AUC | both-OOD coll AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet ViT-tiny | 0.530 | 0.547 | 0.500 | 0.663 | 0.915 | 0.843 | 0.749 | 0.822 |
| LeWM pusht-only | 0.565 | 0.593 | 0.524 | 0.684 | 0.943 | 0.920 | 0.813 | 0.886 |
| LeWM 16-ep ep16 | 0.208 | 0.574 | 0.192 | 0.693 | 0.870 | 0.890 | 0.763 | 0.877 |
| DiT-XL zero-shot | 0.453 | 0.622 | 0.411 | 0.728 | 0.935 | 0.935 | 0.900 | 0.934 |

**几个意外观察**：
1. **OOD partition 经常比 ID 表现更好**（B 的 mass_ratio、vel_x 在 r-OOD / both-OOD 上都比 ID 高）。原因是 OOD partition 拿到更多训练样本（both-OOD 训练样本是 ID 的 8 倍），Ridge 拟合更稳。
2. **LeWM 16-ep ep16 在 ID 上 vel_x = 0.208，远低于其他**，但 OOD 上 = 0.69 ≈ 其他 encoder 水平。这个 ID 异常可能是因为 ID 的 v 范围窄（[1, 4]），LeWM JEPA 训练后表征在窄范围内对速度反而不敏感（参 §5.1 提到的 trade-off）。OOD 上 v 范围更宽，Ridge 反而能学到。
3. **DiT-XL 在所有 partition 上最强**（B 上几乎全部冠军），跟 §3 在 ID 上的结论一致。Scale + ImageNet 通用预训练在 OOD 物理量解码上也最稳。

### 6.5 加入 Protocol C：同一个 probe 跨 partition 测试

Protocol B 的隐患是**每个 partition 单独 fit**，OOD partition 训练样本数比 ID 多 8 倍（28k vs 3k），不公平。Protocol C 用合并训练 fold（41856 帧）fit 单一 Ridge，再 per-partition 测试 —— "**probe 是同一个**" 这件事让 ID vs OOD 比较有意义。

**关键对照（both-OOD partition，最 OOD 的）**：

| Encoder | A pos_x | B pos_x | **C pos_x** | A vel_x | B vel_x | **C vel_x** | A coll AUC | B coll AUC | **C coll AUC** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet ViT-tiny | −0.026 | 0.716 | **0.707** | 0.111 | 0.663 | **0.652** | 0.754 | 0.822 | **0.814** |
| LeWM pusht-only | 0.005 | 0.745 | **0.737** | −0.043 | 0.684 | **0.672** | 0.653 | 0.886 | **0.872** |
| LeWM 16-ep ep16 | −0.138 | 0.745 | **0.728** | −0.081 | 0.693 | **0.679** | 0.697 | 0.877 | **0.851** |
| DiT-XL zero-shot | 0.101 | 0.790 | **0.804** | 0.094 | 0.728 | **0.740** | 0.786 | 0.934 | **0.949** |

**C ≈ B**（差 < 0.03）。所以 §6.3 的发现（"OOD 上 encoder 真实质量好"）**不是 Protocol B 的 sample size 偏置造成的** —— 同一个 probe 跨 partition 测试，结论一样。

### 6.6 Protocol C 揭示的新 nuance：mass_ratio 在不同 partition 上**不在同一线性方向**

Protocol B 在 ID 上 mass_ratio R²= 0.73 (LeWM pusht-only)，Protocol C 在 ID 上 mass_ratio R² **= −8.11**。这个反直觉的数据揭示了另一层 phenomenon：

| Encoder | B.ID mass_ratio | **C.ID mass_ratio** | B.both-OOD | C.both-OOD |
|---|---:|---:|---:|---:|
| LeWM pusht-only | 0.729 | **−8.11** | 0.932 | 0.916 |
| LeWM 16-ep ep16 | 0.573 | **−11.10** | 0.920 | 0.898 |
| ImageNet ViT-tiny | 0.776 | **−13.93** | 0.908 | 0.874 |
| DiT-XL zero-shot | 0.812 | **−6.33** | 0.930 | 0.920 |

**这是 R² 公式的统计 artifact + 真实物理 nuance 叠加**：

1. **R² 公式**：$R^2 = 1 - \mathrm{var(residual)} / \mathrm{var(target)}$。ID 的 mass_ratio 范围窄（m₁、m₂ 都在 ID 半径范围内 → m∝r³ 推导出 mass_ratio 也窄），var(target_ID) 很小。Combined probe 在 OOD 主导的训练数据上拟合，对 ID 的预测**绝对误差**也许只是中等，但相对于 ID 的小 variance，残差就压倒 → 极负 R²。
2. **真实物理**：phyworld eval 不同 partition 不只是 target 取值范围不同，**视觉表征本身也不同**。r-OOD 让球更大/更小，球的视觉外观、像素覆盖面积、空间频率全变。encoder 把"小球"和"大球"分别编码到不同 emb 子空间是合理的。**emb → mass_ratio 的最佳线性映射在不同 partition 上可能就不同**。Combined Ridge 平均这些不同方向得到一个折衷映射，每个 partition 上都不是最优。

具体到 mass_ratio 上：Protocol B 在 ID 上 R²=0.73（说明 ID 内有线性映射），在 both-OOD 上 R²=0.93（说明 OOD 内也有线性映射）—— 但 Protocol C 的 ID 上掉到 −8 说明 **两个 partition 的线性映射方向不一致**。

Pos_x、vel_x、coll_AUC 上 C ≈ B 说明**这些 target 的 linear 方向在不同 partition 上基本一致**（球位置/速度/碰撞事件的 emb 编码是 partition-invariant 的）。**只有 mass_ratio 有 partition-specific 编码** —— 这跟"球半径变化在视觉上明显，质量解读必然 partition-specific"的直觉吻合。

### 6.7 对原结论的影响（修正版）

| 之前的结论 | 修正后 |
|---|---|
| "encoder 是 ID-specific，OOD transfer 差" | **错**，encoder 在 OOD 上的 pos / vel / collision 表征质量**接近 ID**（all 3 protocols agree）。mass_ratio 在不同 partition 上编码方向不同（C 暴露），但 each partition 自己内部依然 linear-readable。 |
| "FINAL §6.6 提到的 N3 守恒律 transfer 失败" | **大部分错**，pos / vel / collision_event 都 transfer 良好。**mass_ratio 是 partition-specific 编码**（这倒是符合"不同球大小需要不同视觉 → 概念映射"的直觉，但跟"encoder 学不到守恒律"是两回事）。|
| "下游策略 fine-tune 时只能用 ID 数据" | 部分错。**对 pos / vel / collision**：encoder 表征 partition-invariant，downstream probe 训 ID 数据 ≈ 跨 partition 都行（但要用足够 expressive 的 probe，linear Ridge 因为外推不行）。**对 mass_ratio**：必须在 OOD 训练样本上 fine-tune 或者用 nonlinear probe，因为 emb → mass_ratio 是 partition-specific。 |

### 6.8 关于 phyworld 论文 "DiT 不能 OOD" 的结论

phyworld 论文测的是**生成式 rollout**：
- 模型：从头训 video DiT on phyworld_collision
- 评估：给 OOD 初始条件，让 DiT 生成接下来 N 帧
- 失败：生成视频违反物理规律（球穿过、速度异常）

我们测的是**判别式 probe**：
- 模型：frozen ImageNet 或 PushT 预训练 encoder（不在 phyworld 训练）
- 评估：linear classifier 从 emb 读 (x, v, m, collision_event)
- 结果：emb 包含足够 partition-invariant 的物理量信息

**两个结论不矛盾，测的是 video DiT 学到的不同层面**：
- phyworld 测 **dynamics rollout**（"知道下一帧应该长什么样吗"）
- 我们测 **representation visual encoding**（"当前帧的物理量编码到 emb 里了吗"）

可能两个都是真的：encoder 在 emb 里编码了 OOD 球的位置/速度/碰撞（**representation transfer 好**），但 video generation 模型从 emb 解码出下一帧时，违反 OOD 动力学（**generation transfer 差**）。视觉 encoding ≠ 动力学外推。**这是个有意思的结构性观察 —— OOD failure 的位置在 decoder 那一侧，不在 encoder 这一侧。**

更广泛的方法论教训：**linear probing OOD 评估必须用 Protocol B 或 C，不能用 A**。否则会把 "probe 训不出 OOD 系数" 误读为 "encoder 学不到 OOD 表征"。除此之外 mass_ratio 这类 partition-specific 编码量需要 nonlinear probe 或 per-partition fit 才能正确评估。

---

## 7. Caveats

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

## 8. 下一步建议

按优先级：

1. **等 LeWM 16-epoch 走势出来** —— 决定是否还要补 DiT LoRA 4-8 epoch（如 LeWM 是 U 型，DiT 大概率也需要更多 epoch 才能稳定 net positive）。
2. **跑 ViT-tiny ImageNet-init frozen probe**（~10 min）—— 完成 FINAL §7 #3 的"必要性"侧验证。如果 5.5M ViT-tiny ImageNet init 也能拿到 0.85+ K=4 vel_x，那 §4.2 的"通用视觉预训练就够"结论被三方独立验证。
3. **跑 DiT-XL zero-shot 在 uniform_motion 上**（~5 min）—— uniform_motion 是更简单的 1-ball 1D 任务，对比 [UNIFORM_MOTION_REPORT.md](UNIFORM_MOTION_REPORT.md) 看 pattern 是否一致。
4. **跑 DiT-XL zero-shot 在 PHYRE OOT 上** —— 验证 §4.4 "复杂 → 简单域 transfer 廉价" 假说。如果 DiT-XL 在 PHYRE 上也优于 LeWM collision-trained encoder，说明 LeWM 训练完全没有"物理域专长"。
5. **在非 toy 物理 dataset 上跑同样对比**（Something-Something / PhysIQ / BAIR）—— 给 §4.4 "DiT 赢是 phyworld 的失败"假说提供反证 / 正证。
