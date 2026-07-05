# Probe vs 固定编码位置（Structured Slot）效果汇总

**日期**：2026-07-03
**实验域**：`uniform_motion`（部分速度可解码性对比含 `collision`）
**目的**：横向对比两条"结构化 latent"路线 —— probe loss（可读出物理量）与固定编码位置 structured slot（强制指定 latent 维度=物理量）—— 各自对 rollout、OOD 泛化、状态可读性带来的效果。
**数据来源**：
- 固定编码位置主结果：[structured_loss_report.md](structured_loss_report.md)
- probe 背景结果：[../6-2/piwm_uniform_collision_results.md](../6-2/piwm_uniform_collision_results.md)

---

## 0. TL;DR

| 路线 | 一句话效果 | 适用/风险 |
|---|---|---|
| **probe** | 保住"状态可读 + 长程一致"，速度可解码性大幅提升 | 稳定净正向，尤其长 horizon 和 r/m-OOD |
| **固定编码位置** | 提升"短程局部对齐 + 可解释性"，ID/短程 pixel 最强 | 长程失效、both-OOD 退化（`weight=1.0` 过硬） |

**核心判断**：两者解决的是不同问题，且**都没解决长程漂移的根因** —— 因为它们只约束"状态是什么"，没约束"状态怎么变"。下一步应从状态对齐走向 latent dynamics 约束。

---

## 1. 三种机制的区别

| 名称 | 机制 | 对 latent 的约束强度 |
|---|---|---|
| **baseline** | 只做 next-latent 预测 | 无 |
| **probe**（posonly_probe） | `probe_head(emb) ≈ proprio`，额外 head 读出物理量 | 弱：latent 不需固定维度对齐，只要"能被读出" |
| **固定编码位置**（structpos） | `emb[..., 0:2] ≈ proprio`，强制前两维=位置 | 强：直接指定 slot 承载物理量 |

一句话对比：probe 是"**能读出**位置"，固定编码位置是"**规定哪几维就是**位置"。后者更可解释，但约束更硬。

---

## 2. Probe 的效果（正向、稳定）

Probe 是目前**更稳的正向增益**，尤其在长程一致性与速度可解码性上。

### 2.1 速度可解码性大幅提升（uniform vx 解码 ρ，K=4）

| partition | baseline | probe(pos-only) |
|---|---:|---:|
| ID | +0.440 | **+0.804** |
| r/m-OOD | +0.564 | **+0.737** |
| v-OOD | +0.782 | **+0.944** |
| both-OOD | +0.893 | **+0.928** |

四个 partition 全面优于 baseline。

### 2.2 latent cos 全程更好，且长 horizon 不塌（uniform）

| h | baseline | probe(pos-only) |
|---|---:|---:|
| 1 | +0.9924 | +0.9940 |
| 8 | +0.9119 | +0.9316 |
| 16 | +0.8445 | **+0.8653** |
| 28 | +0.8432 | **+0.8816** |

长程（h=16/28）probe 是最优臂 —— 这是 probe 相对固定编码位置最关键的优势。

### 2.3 r/m-OOD 泛化改善明显（latent cos by partition）

r/m-OOD：`0.7642 → 0.8846`。

**小结**：probe 让 latent 中"物理量可被稳定读出"，对长程一致性和速度信息保留是净正向。

---

## 3. 固定编码位置（structpos）的效果（短程强、长程失效）

效果**两极分化**。

### 3.1 ✅ 正向（短程 / ID / 可读性）

- structured_loss `~2.53 → 0.015`，slot 对齐成功；`pred_loss=0.0035` 未被拖垮，约束与预测兼容。
- 静态 decoder PSNR 仍 **33–35 dB**（ID 34.61 / r-m-OOD 33.76 / v-OOD 35.12 / both-OOD 34.75），未丢视觉信息。
- **短程 pixel rollout PRED PSNR 明显最好**：

| h | baseline | probe | structpos |
|---|---:|---:|---:|
| 1 | 26.49 | 26.90 | **28.17** |
| 2 | 24.18 | 24.90 | **25.62** |
| 4 | 22.27 | 23.23 | **23.38** |

- **ID pixel PSNR 最高**：`24.37 → 25.73`（+1.36 vs baseline，+1.02 vs probe）。

### 3.2 ❌ 负向 / 无收益（长程 / 组合 OOD）

- **长程优势消失甚至倒挂**：

| h | baseline | probe | structpos |
|---|---:|---:|---:|
| 8 | 20.59 | 21.15 | 20.92 |
| 16 | 20.57 | 20.59 | **20.49** |
| 28 | 20.64 | 19.93 | **19.49** |

- **both-OOD 全面退化**：pixel PSNR `20.01 → 19.36`；latent nMSE `0.2948 → 0.4068`。
- **速度演化没变好**：velocity ρ 在 ID / r-m-OOD / both-OOD 都没超过 probe。

**小结**：固定编码位置改善局部状态对齐，但没约束时间演化，误差仍在自回归 rollout 中积累。

---

## 4. 横向对比总表

| 维度 | baseline | probe | 固定编码位置 |
|---|:--:|:--:|:--:|
| 状态可读 / 速度可解码 | 差 | **强** | 中（不如 probe） |
| 短程 pixel rollout (h≤4) | 中 | 中偏好 | **最强** |
| 长程 pixel rollout (h≥16) | 基准 | **最稳** | **最差** |
| ID pixel PSNR | 24.37 | 24.71 | **25.73** |
| both-OOD 泛化 | 基准 | 略好 | **退化** |

### 关键量化对照（vs baseline）

| 指标 | probe | structpos |
|---|---|---|
| ID pixel PSNR | +0.34 | **+1.36** |
| h=1 pixel PSNR | +0.41 | **+1.68** |
| h=28 pixel PSNR | −0.71 | **−1.15** |
| both-OOD latent nMSE | +0.047(变差) | **+0.112(明显变差)** |
| vx 解码 ρ (ID) | **+0.36** | 中间（未超 probe） |

---

## 5. 综合判断

1. **probe 与固定编码位置解决不同问题，优势互补**：probe 保住"状态可读 + 长程一致"；固定编码位置提升"短程局部对齐 + 可解释性"，但牺牲长程与组合 OOD。
2. **两者都没触及长程漂移根因**：它们只约束"状态是什么"，没约束"状态怎么变"。structpos 因约束过硬（`weight=1.0`），在 both-OOD 上反而低于 baseline。
3. **可安全写进论文的表述**：
   > 把位置物理量显式绑定到 latent 固定维度，可提升短程预测与状态可解释性；但长程自回归 rollout 与组合 OOD 下收益消失甚至退化 —— 说明仅做状态对齐不够，物理一致性还需对 latent dynamics 加约束。
   >
   > *Structured slot improves state alignment, but does not by itself enforce physically consistent temporal evolution.*

---

## 6. 下一步建议

1. **structured weight sweep**：`weight ∈ {0.1, 0.3, 1.0}`。`1.0` 大概率过强，先看能否保住短程收益的同时救回 both-OOD / 长程。判断标准同时看 `pred_loss` / latent rollout cos-nMSE / pixel rollout by horizon / both-OOD PSNR，不要只看 `structured_loss`。
2. **加轻量 dynamics loss（更贴主线）**：对位置 slot 的有限差分速度做一致性
   ```
   pred_delta   = z_pos(t+1) - z_pos(t)
   target_delta = proprio(t+1) - proprio(t)
   L_dyn = MSE(pred_delta, target_delta)
   ```
   有显式速度时进一步：`z_pos(t+1) ≈ z_pos(t) + v(t)·dt`。这是从"约束状态"走向"约束演化"的关键一步。
3. **probe + 固定编码位置组合消融**：probe 管长程一致、structpos 管短程/可读，优势互补，做一个 2×2 消融确认能否叠加（尤其看能否同时拿到 structpos 的短程增益和 probe 的长程稳定）。

---

## 7. 产物与日志索引

| 类型 | 路径 |
|---|---|
| structured ckpt | `/data1/likun-share/junjxu/.stable_worldmodel/uniform_motion_structpos_id1k/uniform_motion_structpos_id1k_epoch_20_object.ckpt` |
| latent rollout log | `/data1/likun-share/junjxu/runs/structured_eval/rollout_uniform_structpos.log` |
| pixel rollout json | `/data1/likun-share/junjxu/runs/pixel_rollout_structured/pxroll_structpos.json` |
| 固定编码位置详报 | [structured_loss_report.md](structured_loss_report.md) |
| probe 详报 | [../6-2/piwm_uniform_collision_results.md](../6-2/piwm_uniform_collision_results.md) |
