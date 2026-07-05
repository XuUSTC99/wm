# Structured Latent Slot Loss — uniform_motion 初步结果

**日期**：2026-07-03  
**实验域**：`uniform_motion`  
**目的**：先不引入显式 dynamics model，只测试 structured loss：把 latent 中固定位置的若干维直接约束为物理量，观察它是否能缓解 rollout 漂移和 OOD 退化。

---

## 0. TL;DR

这次 structured loss 在机制上是生效的：`emb[..., 0:2]` 被拉到 `proprio` 位置量附近，最终 `validate/structured_loss ≈ 0.0148`。但是从 rollout 结果看，它不是一个稳定的正向提升：

| 现象 | 结论 |
|---|---|
| 静态 decoder PSNR | structured latent 仍然很好解码，ID/OOD 都在 33-35 dB |
| pixel rollout 短程 | h=1/h=2 明显优于 baseline 和 probe |
| pixel rollout 长程 | h=16/h=28 没有提升，h=28 反而低于 baseline/probe |
| OOD rollout | r/m-OOD 小幅提升，但 both-OOD 明显下降 |
| latent rollout | ID 接近 baseline；r/m-OOD nMSE 稍好；v-OOD/both-OOD 不稳定 |

**主结论**：固定 latent slot 可以让状态表示更可解释、更容易读出位置，但它本身没有约束时间演化规律，所以不能单独解决长程漂移。下一步更合理的是在 structured slot 上加 dynamics loss，或者做较小权重 sweep，而不是直接把 `structured.weight=1.0` 当成最终方案。

---

## 1. 方法设置

### 1.1 和 probe loss 的区别

之前的 probe loss 是：

```text
probe_head(emb) ≈ proprio
```

即物理量由一个额外 probe head 从 latent 中读出。latent 本身不需要在固定维度上对应物理量。

这次 structured loss 是：

```text
emb[..., 0:2] ≈ proprio
```

也就是强行指定 latent 的前两维承担位置物理量。这样做的假设是：如果状态表示中有一部分维度直接对齐物理状态，predictor 可能更容易学习物理演化。

### 1.2 训练配置

核心配置：

```yaml
data: phyworld_uniform_motion_id1k
trainer.max_epochs: 20
loss.probe.weight: 0.0
loss.structured.weight: 1.0
loss.structured.target: proprio
loss.structured.start_dim: 0
init_from_ckpt: /data1/likun-share/junjxu/.stable_worldmodel/lewm_paper_pusht/weights.pt
```

对应 checkpoint：

```text
/data1/likun-share/junjxu/.stable_worldmodel/uniform_motion_structpos_id1k/uniform_motion_structpos_id1k_epoch_20_object.ckpt
```

实现改动：

- [`le-wm/config/train/lewm.yaml`](../../le-wm/config/train/lewm.yaml)：新增 `loss.structured` 配置，默认 `weight=0.0`
- [`le-wm/train.py`](../../le-wm/train.py)：新增 structured slot MSE，只有 `loss.structured.weight > 0` 时生效

---

## 2. 训练 sanity check

最终 epoch 验证指标：

| metric | value |
|---|---:|
| `validate/pred_loss` | 0.003522 |
| `validate/structured_loss` | 0.014753 |

训练过程里 `structured_loss` 从初始约 2.53 快速降到 0.02 以下，说明 latent 前两维确实学到了位置约束。`pred_loss` 最终也没有炸掉，说明这个约束和 next-latent prediction 在训练 loss 层面是兼容的。

但这里仍然只能说明“slot 对齐成功”，不能说明 rollout 物理变好。下面的 rollout 指标才是关键。

---

## 3. 静态 universal decoder

structured 模型单独训练了对应的 proj-space universal decoder：

```text
/data1/likun-share/junjxu/runs/decoder_viz/universal_proj/udecoder_structpos.pt
```

静态重建 PSNR：

| partition | PSNR |
|---|---:|
| ID | 34.61 |
| r/m-OOD | 33.76 |
| v-OOD | 35.12 |
| both-OOD | 34.75 |

这说明 structured latent 没有丢掉基本视觉信息，至少从单帧 latent 到 pixel 的解码仍然是高质量的。也就是说，后面 pixel rollout 变差不能简单归因于 decoder 不会解码，而更可能来自 autoregressive rollout 本身的状态漂移。

---

## 4. Latent rollout 结果

### 4.1 latent cos / nMSE

| model | ID cos / nMSE | r/m-OOD cos / nMSE | v-OOD cos / nMSE | both-OOD cos / nMSE |
|---|---:|---:|---:|---:|
| baseline | +0.9700 / 0.0647 | +0.7638 / 0.3852 | +0.9313 / 0.1497 | +0.8441 / 0.2948 |
| posonly_probe | +0.9578 / 0.0888 | +0.8306 / 0.3851 | +0.9368 / 0.1536 | +0.8535 / 0.3413 |
| structpos | +0.9661 / 0.0709 | +0.8137 / 0.3597 | +0.9252 / 0.1753 | +0.8253 / 0.4068 |

解读：

- ID：structured 接近 baseline，优于 posonly_probe，但不是明显超过 baseline。
- r/m-OOD：structured 的 nMSE 比 baseline/posonly_probe 稍好，但 cos 不如 posonly_probe。
- v-OOD：structured 比 baseline/probe 都略差。
- both-OOD：structured 明显退化，nMSE 到 0.4068。

### 4.2 K=4 velocity rho

| model | ID | r/m-OOD | v-OOD | both-OOD |
|---|---:|---:|---:|---:|
| baseline | +0.497 | +0.647 | +0.774 | +0.885 |
| posonly_probe | +0.702 | +0.750 | +0.929 | +0.877 |
| structpos | +0.595 | +0.653 | +0.927 | +0.835 |

structured 在 v-OOD velocity rho 上接近 posonly_probe，但 ID / r/m-OOD / both-OOD 都没有超过 probe。这个结果说明固定位置 slot 不等于自然学出更好的速度演化；速度相关信息可能仍然需要显式 dynamics 约束或更合适的监督目标。

---

## 5. Pixel rollout 结果

### 5.1 按 partition 的 PRED PSNR

| model | ID | r/m-OOD | v-OOD | both-OOD |
|---|---:|---:|---:|---:|
| lam0 baseline | 24.37 | 18.50 | 23.64 | 20.01 |
| lam1 probe | 24.71 | 18.54 | 23.60 | 19.88 |
| structpos | 25.73 | 18.83 | 23.52 | 19.36 |

structured 的收益主要集中在 ID 和 r/m-OOD：

- ID：+1.36 dB over baseline，+1.02 dB over probe
- r/m-OOD：+0.33 dB over baseline，+0.29 dB over probe
- v-OOD：略低于 baseline/probe
- both-OOD：明显低于 baseline/probe

这说明 structured slot 对训练分布附近的短期位置预测有帮助，但在半径/质量和速度同时变化时没有提供更强泛化。

### 5.2 按 horizon 的 PRED PSNR

| model | h=1 | h=2 | h=4 | h=8 | h=16 | h=28 |
|---|---:|---:|---:|---:|---:|---:|
| lam0 baseline | 26.49 | 24.18 | 22.27 | 20.59 | 20.57 | 20.64 |
| lam1 probe | 26.90 | 24.90 | 23.23 | 21.15 | 20.59 | 19.93 |
| structpos | 28.17 | 25.62 | 23.38 | 20.92 | 20.49 | 19.49 |

这是最关键的表。structured 在 h=1/h=2/h=4 有明显优势，但随着 rollout 拉长，优势消失：

- h=1：structured 最高，说明下一步预测更准。
- h=2/h=4：仍有收益，但差距变小。
- h=8：低于 probe。
- h=16：低于 baseline/probe。
- h=28：三者中最低。

这和“长程漂移”主线一致：structured slot 能改善局部状态对齐，但如果 predictor 的时间演化没有被物理规律约束，误差仍会在 autoregressive rollout 中积累。

---

## 6. 当前判断

### 6.1 可以引用的正向结果

可以把这次实验作为一个中间证据：

> 将位置物理量显式绑定到 latent 的固定维度，可以提升物理状态的可读性，并改善 ID/短程 rollout 的像素预测质量。

对应证据是：

- `structured_loss` 成功下降到 0.015 左右；
- 静态 decoder PSNR 仍保持 33-35 dB；
- ID pixel rollout 从 24.37 提升到 25.73；
- h=1/h=2/h=4 PRED PSNR 均高于 baseline。

### 6.2 不能过度声称的部分

不能写成“structured slot 解决了物理规律理解不足”。

更准确的表述是：

> Structured slot improves state alignment, but does not by itself enforce physically consistent temporal evolution.

理由是：

- h=16/h=28 没有提升；
- both-OOD pixel rollout 从 baseline 20.01 降到 19.36；
- latent both-OOD nMSE 从 baseline 0.2948 升到 0.4068；
- velocity rho 没有稳定超过 posonly_probe。

---

## 7. 下一步建议

### 7.1 先做 structured weight sweep

当前 `structured.weight=1.0` 可能过强。建议补：

```text
structured.weight ∈ {0.1, 0.3, 1.0}
```

判断标准不要只看 `structured_loss`，要同时看：

- `validate/pred_loss`
- latent rollout cos / nMSE
- pixel rollout by horizon
- both-OOD PRED PSNR

### 7.2 引入 dynamics loss

这次结果支持一个更强的动机：只约束“状态是什么”不够，还要约束“状态怎么变”。

可以在 structured slot 上加一个轻量 dynamics loss，例如对位置 slot 的有限差分速度做一致性：

```text
z_pos(t)      = emb_t[..., 0:2]
z_pos(t + 1)  = emb_{t+1}[..., 0:2]
pred_delta    = z_pos(t + 1) - z_pos(t)
target_delta  = proprio(t + 1) - proprio(t)

L_dyn = MSE(pred_delta, target_delta)
```

如果数据里有显式速度，也可以进一步约束：

```text
z_pos(t + 1) ≈ z_pos(t) + v(t) * dt
```

这个方向比单纯固定 slot 更贴合论文主线：当前 WM 的问题不是完全看不到物体，而是 latent 状态和 transition dynamics 没有被组织成有利于物理演化的形式。

### 7.3 论文表述建议

目前可以写成：

> Preliminary results show that explicitly assigning physical quantities to fixed latent dimensions improves short-horizon prediction and state interpretability, but the benefit diminishes under long autoregressive rollouts and combined OOD shifts. This suggests that structured state alignment alone is insufficient; physical consistency also requires constraints on latent dynamics.

中文意思：

> 初步结果表明，把物理量显式绑定到 latent 的固定维度，可以提升短程预测和状态可解释性；但在长程自回归 rollout 和组合 OOD 条件下收益会消失甚至退化。这说明仅做状态对齐还不够，物理一致性还需要对 latent dynamics 加约束。

---

## 8. 原始日志和产物

| 类型 | 路径 |
|---|---|
| structured ckpt | `/data1/likun-share/junjxu/.stable_worldmodel/uniform_motion_structpos_id1k/uniform_motion_structpos_id1k_epoch_20_object.ckpt` |
| latent rollout log | `/data1/likun-share/junjxu/runs/structured_eval/rollout_uniform_structpos.log` |
| decoder train log | `/data1/likun-share/junjxu/runs/structured_eval/train_udecoder_structpos.log` |
| universal decoder | `/data1/likun-share/junjxu/runs/decoder_viz/universal_proj/udecoder_structpos.pt` |
| static decoder json | `/data1/likun-share/junjxu/runs/decoder_viz/universal_proj/upsnr_structpos.json` |
| pixel rollout log | `/data1/likun-share/junjxu/runs/structured_eval/pixel_rollout_structpos.log` |
| pixel rollout json | `/data1/likun-share/junjxu/runs/pixel_rollout_structured/pxroll_structpos.json` |
