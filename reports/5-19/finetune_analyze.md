# 为什么在 phyworld collision 上做 SSL 微调反而让 probe 变差

**日期**：2026-05-19
**配套实验报告**：[DIT_REPORT.md](DIT_REPORT.md)
**背景**：在 [DIT_REPORT §3-5](DIT_REPORT.md) 里观察到三个 SSL 微调路径都是 net-negative 这个反直觉现象。本文档**专门解释为什么**，包括代码怎么工作、有没有 bug、机制是什么。

---

## TL;DR

> 三个 SSL fine-tune 实验（LeWM-from-scratch、LeWM paper-init+collision-8ep、DiT-XL LoRA+collision-2ep）**全部让 linear probe 表现变差**。代码**没有 bug**，问题是结构性的：
>
> 1. SSL loss（JEPA 预测下一帧 emb / DDPM 预测加进去的 noise）**跟"emb 对 pos/vel/mass 线性可读"在数学上没有对齐**。
> 2. phyworld 视觉太简单，预训练 encoder 已接近 ceiling，**任何微调几乎只能往下走**。
> 3. JEPA 在 16 epoch 后完美 plateau（稳定 trade-off：speed/mass↑ vs pos/coll↓）；DiT LoRA 不收敛、震荡，最终在 epoch 8 因 gradient 爆炸 NaN（缺 gradient clipping）。

---

## 1. 现象总结

[DIT_REPORT §3.4](DIT_REPORT.md) 的 8-way 对比表（K=4，no-projector）：

| Encoder | params | phyworld FT | pos_x | vel_x | mass_ratio | coll AUC |
|---|---:|---:|---:|---:|---:|---:|
| LeWM from-scratch | 5.5 M | 8 ep | 0.814 | 0.594 | 0.666 | — |
| LeWM paper-init + 8 ep | 5.5 M | 8 ep | 0.911 | 0.883 | 0.826 | 0.952 (K=1) |
| **LeWM pusht-only (frozen)** | 5.5 M | **0 ep** | **0.931** | 0.878 | **0.886** | **0.982** |
| **DiT-XL zero-shot** | 749.8 M | **0 ep** | 0.919 | **0.890** | 0.842 | **0.990** |
| DiT-XL + LoRA 2 ep | 749.8 M | 2 ep | 0.907 | 0.829 | 0.841 | 0.971 |

**模式**：所有冠军 cell 都在 **frozen 预训练**（0 ep）encoder 上，**没有任何 fine-tune 方案进过冠军榜**。同模型对比：

- LeWM pusht-only (0ep) **>** LeWM paper-init+8ep（fine-tune 让 mass_ratio −0.06, coll AUC −0.03）
- DiT zero-shot **>** DiT + LoRA 2 ep（fine-tune 让 pos_x −0.012, vel_x −0.061, coll AUC −0.019）

---

## 2. 微调代码怎么工作的

### 2.1 LeWM JEPA 微调（[le-wm/train.py](../../le-wm/train.py)）

核心 4 行（[le-wm/train.py:39-42](../../le-wm/train.py)）：

```python
ctx_emb = emb[:, :ctx_len]                       # 前 ctx_len 帧的 emb
tgt_emb = emb[:, n_preds:]                       # 第 n_preds 帧之后的 emb（label）
pred_emb = self.model.predict(ctx_emb, ctx_act)  # 用前面预测后面
loss = (pred_emb - tgt_emb).pow(2).mean() + 0.09 * sigreg
```

**优化目标**："emb 序列必须可预测" + SIGReg 防止表征 collapse。

数据流：
```
pixels[t] ─→ encoder (ViT-tiny) ─→ cls token ─→ projector ─→ emb[t]
                                                              ↓
emb[1:t], action[1:t] ─→ predictor (6-layer ARPredictor) ─→ pred_emb[t+1]

pred_loss = MSE(pred_emb[t+1], emb[t+1])
sigreg_loss = SIGReg(emb)   # 防止 emb collapse 到常量
```

可训参数（约 18M）：
- encoder ViT-tiny（5.5M）✓
- projector / pred_proj MLP ✓
- predictor ARPredictor ✓
- action_encoder ✓

**关键 trade-off**：loss 鼓励 emb 序列 smooth / 可预测，**不要求** emb 包含 pos/vel/mass 等 linear-readable 信息。如果某个特征对预测下一帧没帮助（如球的绝对位置 —— 相对位置和 velocity 已足以预测），梯度会**压缩掉它**。

### 2.2 DiT LoRA 微调（[phyworld/scripts/dit_lora_finetune_probe.py](../../phyworld/scripts/dit_lora_finetune_probe.py)）

核心 6 行（[dit_lora_finetune_probe.py:198-213](../../phyworld/scripts/dit_lora_finetune_probe.py)）：

```python
latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor   # (B,4,32,32)
t = torch.randint(0, 1000, (B,))                                       # 随机 timestep
noise = torch.randn_like(latent)
noisy = scheduler.add_noise(latent, noise, t)                          # 加噪
pred = transformer(noisy, t, class_labels=1000)[0]                     # DiT 去噪
loss = F.mse_loss(pred[:, :4], noise)                                  # 标准 DDPM
```

**优化目标**：标准 DDPM noise prediction —— 给一个 timestep `t`，模型预测加进去的 Gaussian noise。

数据流：
```
image[t] ─→ VAE encode ─→ latent (4×32×32)
                          ↓ + scheduler.add_noise(t)
                          noisy_latent
                          ↓
                          DiT-XL transformer (frozen base + LoRA on last 4 attn blocks)
                          ↓
                          predicted noise (4×32×32)

loss = ||predicted_noise - actual_noise||²
```

可训参数（0.59M / 750M total = 0.079%）：
- LoRA adapters on **last 4** transformer blocks 的 `attn1.{to_q, to_k, to_v, to_out.0}`，rank=16, alpha=32
- **base 750M 参数全 frozen**
- VAE 全 frozen

**关键 trade-off**：loss 鼓励 attention **专门擅长去噪 phyworld 图像**（圆球、纯色背景）。LoRA 把 last-block 的 attention representation 朝"狭窄的 phyworld-only 去噪最优"漂移，**牺牲了 ImageNet 上学到的通用视觉结构**（这恰好是 probe 友好的部分）。

---

## 3. 有没有 bug？逐条检查

| 嫌疑 | 检查结果 | 是 bug 吗 |
|---|---|---|
| LeWM target 没 stop_grad | `tgt_emb = emb[:, n_preds:]` 没 detach；但配合 SIGReg 防 collapse，是 lewm 论文的 JEPA 变体设计 | **不是 bug**，是设计选择 |
| DiT LoRA 加载用 `load_state_dict(strict=False)` | inference 时 silent fail 不应用 LoRA（导致 NaN inference） | **是 bug**，已修（用 `PeftModel.from_pretrained + merge_and_unload`） |
| DiT 训练没 gradient clipping | epoch 8 LoRA weights 爆 NaN | 不算训练 bug，但 DDPM loss 抖动大时必须加 `clip_grad_norm_(1.0)` |
| `class_labels=1000` (null) | DiT-XL 训练时 0-999 为 ImageNet 类，1000 为 null。这是正确的"无类别条件"用法 | **不是 bug** |
| LoRA 只加在 attn1 上 | 标准 LoRA 配方（ff 层没加） | 不是 bug，是 capacity 选择 |
| VAE `scaling_factor=0.18215` | SD VAE 标准值，diffusers 自动从 config 读 | **不是 bug** |

**结论**：没有让 fine-tune 结果变差的 bug。变差是**结构性的**。

---

## 4. 为什么"结构性地"变差

五个机制按贡献度排序：

### (A) Catastrophic forgetting 通用视觉特征 — 主因

预训练 encoder（DiT-XL on ImageNet 1000 类 / lewm-pusht on PushT）学到了**极丰富的通用视觉**：边缘、纹理、颜色、形状、各种语义概念。这些通用特征恰好涵盖了 phyworld probe 需要的（ball position, radius, color contrast）。

phyworld 微调让 encoder 朝"只擅长 phyworld 的小球+黑底"方向塌缩。**对 probe 有用的通用特征被部分 catastrophic forgetting 抹掉**。

直接证据：
- DiT LoRA 后 **mass / mass_ratio 几乎不变**（静态视觉量，每帧都需要，被反复"使用"，没被忘）
- DiT LoRA 后 **vel_x / accel_x 跌得最狠**（dynamics 量，依赖预训练中学到的"自然图像中的物体运动模式"，phyworld 微调时被覆盖）

### (B) Fine-tune 目标和 probe 目标错位（核心）

| Encoder | Fine-tune loss | Probe objective |
|---|---|---|
| LeWM JEPA | ‖pred(z_t, a_t) − sg(z_{t+1})‖² + SIGReg | linear classifier on emb → physical quantity |
| DiT LoRA | ‖ε − ε_θ(z_noisy, t, y)‖² | 同上 |

两个 fine-tune loss 都**没有约束 emb 必须对 pos/vel/mass 线性可读**。它们优化的是"预测能力"或"去噪能力"，不是"线性可读的物理量编码"。

实际效果：fine-tune 可能让 emb 在某个**对 prediction 有用但对 linear probe 无用**的方向上重组（比如把"下一帧球的轨迹外推"信号增强了，但损失了"当前帧球的 absolute position"的线性可读性）。这种重组对 probe 是负贡献。

### (C) Probe 任务已经在 ceiling 附近

DiT-XL zero-shot K=4 collision AUC = 0.990，pos_x = 0.92。**已经接近 ceiling**。即使 fine-tune 真的学到一些新东西，从 0.92 涨到 0.95 比从 0.92 跌到 0.91 难得多。这是一个**非对称的不稳定平衡**：稍有扰动只能往下走。

### (D) LoRA 0.59 M 参数 × 2 epoch 太少，DDPM loss 抖动太大

DiT LoRA 加 0.59 M 参数 + 2 epoch 总共 4000 step 的 DDPM noise prediction。DDPM loss 单步抖动巨大（实测 0.0069 → 0.1232 → 0.0150 这种量级跳跃），因为 noise scale 跟随机采样的 timestep 强相关。**很可能 LoRA 没学到任何 phyworld-specific 信号，只是在 emb 上加了一层结构化噪声**。

trajectory data 确认（[DIT_REPORT §5.2](DIT_REPORT.md)）：DiT LoRA epoch 2/4/6 在 zero-shot 附近震荡（pos_x 0.907 / 0.915 / 0.907 / vel_x 0.830 / 0.854 / 0.821），不是稳定收敛，是**布朗运动 + drift**。

epoch 8 时 LoRA weight max\|w\| 从 0.31 → 0.44 → 0.58 → **NaN** 训练发散。根本原因是 DDPM loss 抖动大 + AdamW lr=1e-4 + **没启用 gradient clipping**。修复方法：加 `torch.nn.utils.clip_grad_norm_(..., 1.0)`。

### (E) Phyworld 视觉太简单，"新东西可学" 接近 0

phyworld_collision 整张图就是 (黑底 + 2 圆 + 颜色 1-2 种)，信息熵极低。一个 749M 参数的 DiT 几乎**一眼看穿**。fine-tune 阶段模型实际上**没有新概念可以 fit**，gradient 提供的有效信号几乎为 0。所有梯度更新都退化为对现有表征的随机方向扰动。

### 综合归因

(A) + (E) 联合是上游因（"目标域太窄、预训练表征已足够、再训练只能磨损"）。
(B) 是中介机制（"fine-tune 目标本来就没保证对 probe 有用"）。
(C) 解释了"为啥扰动一定让 probe 变差而不是变好"。
(D) 是 DiT 这一个 setup 的额外加速因子。

---

## 5. Trajectory 证据：JEPA 与 DDPM 微调的两种失败模式

### 5.1 LeWM 16-epoch trajectory — 完美 plateau + trade-off

| Encoder | pos_x | vel_x | speed | mass | mass_ratio | accel_x | coll AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| pusht-only (0 ep) | **0.931** | 0.878 | 0.486 | 0.945 | **0.886** | **0.271** | **0.982** |
| 16-ep epoch 4 | 0.909 | 0.873 | 0.532 | 0.952 | 0.849 | 0.245 | 0.975 |
| 16-ep epoch 8 | 0.913 | 0.879 | 0.552 | 0.955 | 0.867 | 0.240 | 0.974 |
| 16-ep epoch 12 | 0.915 | 0.876 | 0.545 | 0.956 | 0.862 | 0.236 | 0.975 |
| 16-ep epoch 16 | 0.913 | 0.875 | **0.542** | **0.956** | 0.864 | 0.234 | 0.974 |

**epoch 4 到 16 完美 plateau**（max delta < 0.01），不是欠拟合假象，是**稳定 trade-off**：

| 类别 | targets | Delta (epoch 16 − pusht-only) |
|---|---|---:|
| **微涨** | speed, mass | +0.06 / +0.01 |
| **持平** | vel_x | −0.003 |
| **微跌** | pos_x, mass_ratio, accel_x, coll AUC | −0.018 ~ −0.037 |

**机制推测**：JEPA 训练目标是预测下一帧 latent，**鼓励 encoder 编码"在接下来会变什么"的信息**。speed = ‖velocity‖ 直接决定下一帧偏移幅度，被 prediction loss 大量放大；而 absolute position、collision boolean 这种"当前帧静态"信号对预测没什么帮助，被弱化。这跟 [5-12/COLLISION_REPORT §2.3 (b)](../5-12/COLLISION_REPORT.md) 提到的 "projector 在压缩对 prediction 无用的信息" 是同一机制，但这次出现在 projector 之前（no-projector probe）—— **encoder 本身在按 prediction 目标重构表征**。

**所以不是"忘掉了"，是"主动地重新分配了 encoder 容量"**。

### 5.2 DiT LoRA 8-epoch trajectory — 不收敛 + 训练发散

| Encoder | pos_x | vel_x | speed | mass | mass_ratio | accel_x | coll AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot (0 ep) | **0.919** | **0.890** | 0.435 | 0.943 | 0.842 | **0.409** | **0.990** |
| LoRA epoch 2 | 0.907 | 0.830 | 0.370 | 0.944 | 0.842 | 0.265 | 0.971 |
| LoRA epoch 4 | 0.915 | 0.854 | 0.354 | 0.947 | **0.869** | 0.294 | 0.975 |
| LoRA epoch 6 | 0.907 | 0.821 | 0.325 | 0.942 | 0.851 | 0.237 | 0.961 |
| LoRA epoch 8 | ❌ **NaN**（训练发散） |

LoRA weight max\|w\| 走势：
- epoch_2: 0.313
- epoch_4: 0.443
- epoch_6: 0.577
- **epoch_8: NaN** ← gradient 爆炸

epoch 2/4/6 在 zero-shot 附近震荡，没有任何收敛迹象。**不是 catastrophic forgetting 稳定下来，是 random walk + drift**。

### 5.3 两种失败模式对照

| 维度 | LeWM JEPA | DiT LoRA |
|---|---|---|
| 训练 loss 性质 | ‖pred − target‖² + SIGReg，稳定 | DDPM noise pred，单步极不稳 |
| epoch 4-16 trajectory | 完美 plateau，方差 < 0.01 | 震荡 + drift，方差 0.02-0.06 |
| 训练终态 | epoch 16 ≈ epoch 4 | epoch 6 < epoch 4，epoch 8 NaN |
| vs zero-shot | 整齐 trade-off：speed+mass↑ / 其余↓ | 全面 net negative，没有 silver lining |

两种完全不同的 pattern 都印证"在 phyworld 上 SSL 微调不能让 probe 更准"，但揭示了**两种不同的失败模式**：
- **JEPA**：稳定但选择性地丢失对 probe 有用的信息
- **DDPM**：根本不稳定，朝着随机方向漂，最终训练发散

---

## 6. SSL loss 的本质问题

**SSL = Self-Supervised Learning** —— 不需要人工标注的训练目标。本实验里的两个 SSL loss：

| Loss | 类型 | "label" 怎么来 |
|---|---|---|
| JEPA `‖pred − emb[t+1]‖²` | 时序自监督 | emb[t+1] 来自下一帧（数据自身） |
| DDPM `‖ε − ε_θ(noisy, t)‖²` | 重建自监督 | ε 是代码 `torch.randn` 自己加的 |

对比：
- **Supervised**（如 ImageNet 分类）：label 是"这张图是什么"，直接对齐"图里有什么" → emb 必然包含识别物体的信息 → probe 友好
- **SSL**（JEPA / DDPM）：label 是"下一帧 emb"/"加的 noise"，跟"图里有什么"**没有直接联系**

**关键洞察**：SSL loss 是否对 probe 友好，取决于 **pretext task 是否隐式包含 probe 想要的信息**。

| SSL 方法 | Pretext task | 是否隐式包含 probe 目标 |
|---|---|---|
| CLIP | 图-文对齐 | ✅ 文本本身含语义 → 图 emb 包含物体类别 |
| SimCLR | 同图增强一致性 | ✅ 同物体不同视角必须 close → emb 编码物体身份 |
| MAE | mask reconstruction | ⚠️ 间接，需要重建被 mask 的 patch → emb 必须含 spatial 信息 |
| **JEPA / DDPM** | 时序预测 / 去噪 | ❌ 不直接，可以靠"压缩到 task-relevant 子空间"满足 loss |

在 phyworld 这种**窄分布**上，JEPA / DDPM 的 loss 的最优解可以是"狭窄的 phyworld-only" 表征，破坏掉原本通用的 probe-friendly 结构。这就是**为什么 fine-tune 几乎必然让 probe 变差**。

---

## 7. 怎么改才能让微调真的有效

| 改法 | 原因 | 预期效果 |
|---|---|---|
| **加 probe-aware aux loss**：JEPA loss + 0.1 × (pos/vel probe loss on emb) | 直接告诉 encoder "保持 pos/vel linear-readable" | 微涨 probe，但跟 SSL 本意冲突 |
| **加 gradient clipping** (clip_grad_norm=1) | 防 DiT LoRA NaN | 让 epoch 8/16 能完成（不一定改变方向） |
| **换 dataset**：phyworld → 真正需要 dynamics 学习的（双摆、流体、3D 多刚体）| (E) 假设的反证 | 可能 fine-tune 真的有正贡献 |
| **训更少 epoch**（DiT 1 ep / LeWM 2 ep）| 在 catastrophic forgetting 开始之前停 | 可能持平不变差 |
| **混合预训练 + phyworld 数据**：每 batch 一半 ImageNet 一半 phyworld | 防 catastrophic forgetting | 通常 0.5-2% 改善（rehearsal learning） |

---

## 8. 教训总结

1. **SSL loss 跟 linear probe accuracy 没有数学保证的对齐**。Probe-friendly 表征是大规模、多样化 SSL 训练的 emergent property，不是 loss 直接驱动。

2. **在窄分布的简单数据上做 SSL 微调几乎必然伤害预训练 encoder**。phyworld 是个很好的例子：视觉信息熵低，loss 优化的方向跟 probe 友好度无关甚至反方向。

3. **"Fine-tune 没让模型变更好"不等于"fine-tune 没干活"**。LeWM JEPA 在 16 epoch 上的 pred_loss 从 0.039 → 0.017 是真的降了，DDPM loss 也真的在降。它们**降的方向跟 probe 友好度无关**。

4. **DDPM-style fine-tune 必须有 gradient clipping**。Loss 抖动大 + AdamW lr 在 1e-4 → weights 朝最近的局部 minima 漂，最终爆炸。

5. **代码没 bug，结论是真的**：reproducibility 高，三个独立路径（LeWM-fromscratch、LeWM paper-init+collision、DiT LoRA）一致结论。
