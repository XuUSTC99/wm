# 6-24 综合终报：物理结构化 World Model 的完整结论

**日期**：2026-07-05
**范围**：把 6-24 目录下 5 份报告的重点汇总成一份。研究主线——**怎么让 le-wm 的 latent world model 在 OOD 和长程 rollout 上变好**，试过 probe、固定编码、二阶动力学、free-rollout、承重编码等一系列手段。

> 源报告（细节见各自文件）：
> - [structured_loss_report.md](../structured_loss_report.md) — 固定编码位置(structpos)初步
> - [probe_vs_structpos_summary.md](../probe_vs_structpos_summary.md) — probe vs 固定编码 对比
> - [diagnostic_report.md](../diagnostic_report.md) — deep-supervision 方法论诊断
> - [piwm_dynamics_conclusion.md](../piwm_dynamics_conclusion.md) — 二阶动力学 + PIWM 对照 + free-rollout 发现
> - [kinematics_exploration.md](../kinematics_exploration.md) — 怎么让运动学真正帮到 OOD/长程

---

## 0. TL;DR（五条核心结论）

1. **真正的主升力是 free-rollout 训练（去掉 teacher forcing）**，不是任何物理结构化手段。三域（uniform/collision/parabola）一致大幅提升，both-OOD nMSE 砍半以上。**已设为默认。**
2. **训练 rollout 长度 num_preds 要按动力学复杂度调**（2026-07-05 新增，见 [optimization_plan §3](optimization_plan.md)）：**collision 甜点≈20**，both-OOD 0.393→**0.294（−25%）**、长程 h28 cos 0.58→**0.70**、ID 3 倍；但 **uniform/parabola 保持 8**（调大反而有害）。冲量/多步域吃长 rollout，简单域不吃。**这是 collision 上首个大提升。**
3. **运动学方程本身不带来提升**——直接挂 2 维 slot + smooth accel MLP，三域净贡献≈0 甚至变差；collision 完全不吃运动学。
4. **但配齐条件后运动学能帮**：`free-rollout + 承重编码(pos_weight) + 光滑动力学域 + pixel 尺` 四条同时满足时净超纯 FR（uniform pixel 长程 **+1.25dB**、parabola both-OOD **0.313→0.262**）。
5. **方法论教训**：latent cos / K=4 ρ 涨 ≠ world model 变好；主指标要用 **pred_loss / pixel**，别信被稀释或被 probe 主导的代理指标。consistency loss 实测未见收益。

**当前各域最优配方**：uniform/parabola = `free-rollout(np8) + 承重 + 运动学`；**collision = `free-rollout + num_preds≈20 + 纯 FR`**。

---

## 1. 研究脉络（试了什么，按顺序）

| 阶段 | 手段 | 结果 |
|---|---|---|
| ① probe | probe head 读出物理量 | 保长程一致 + 速度可解码，但没解决长程漂移 |
| ② 固定编码 structpos | 强制 `emb[:,0:2]=位置` | 短程/ID 变好，长程不变、both-OOD 退化 |
| ③ 二阶动力学 | 位置 slot 走 `z+v+accel` | latent 上无提升甚至变差（accel MLP 过拟合） |
| ④ **free-rollout** | 训练时喂自己的预测（无 teacher forcing） | **三域全部大幅提升 ← 主升力** |
| ⑤ 承重编码 pos_weight | pred_loss 里加重物理 slot | 让运动学在光滑域净超纯 FR（pixel 尺） |
| ⑥ consistency loss | 约束预测速度=真实速度（form-free） | 冲 collision 缺口（用户在验证中） |

---

## 2. probe vs 固定编码（阶段 ①②，源：structured_loss / probe_vs_structpos）

| 维度 | baseline | probe | 固定编码 structpos |
|---|:--:|:--:|:--:|
| 状态可读/速度可解码 | 差 | **强** | 中 |
| 短程 pixel (h≤4) | 中 | 中 | **最强** |
| 长程 (h≥16) | 基准 | **最稳** | **最差** |
| both-OOD | 基准 | 略好 | **退化** |

**结论**：两者都只约束"状态是什么"、没约束"状态怎么变"，所以都没解决长程漂移。这直接引出阶段③④。

---

## 3. free-rollout 是主升力（阶段③④，源：piwm_dynamics_conclusion）

### 3.1 二阶动力学为什么没用
挂 2 维 slot + smooth accel MLP：uniform 上 accel 过拟合（真实 a=0 却学出 ~0.5·|v| 乱修正），v-OOD/长程退化。换纯匀速/正则能修回过拟合但仍不超 baseline。

### 3.2 读 PIWM 论文发现真正的关键
PIWM([2412.12870](../../../papers/2412.12870.pdf)) §4.1 明确：**全程 free-rollout 训练、反对 teacher forcing**（teacher forcing 掩盖误差累积→部署时长程崩）。le-wm 原本是纯单步 teacher-forced ← 长程漂移头号根因。

### 3.3 三域实证（latent both-OOD nMSE↓）

| 域 | teacher-forced | **free-rollout** |
|---|---|---|
| uniform | 0.308 | **0.131** |
| collision | 1.114 | **0.393** |
| parabola | 0.786 | **0.313** |

free-rollout 三域全部大幅提升；长程 cos 普遍 +0.3 以上。**这是整条研究线里唯一稳定、通用、无副作用的净提升，已设为默认（`wm.free_rollout=true, num_preds=8, batch=64`）。**

---

## 4. 怎么让运动学真正帮 OOD/长程（阶段⑤，源：kinematics_exploration）

### 4.1 关键洞察：解码位置 ρ（绕开 192 维稀释）
运动学恰好在 **v-OOD（速度分布外）把位置守到 0.99**（匀速外推对任意速度精确），但 **r/m-OOD（外观变）崩到 0.29**（外观 shift 破坏 slot 编码）。→ 问题不在动力学，在**物理通道非承重 + 编码对外观不鲁棒**。

### 4.2 承重编码（pos_weight）激活运动学价值

**uniform pixel by-horizon（PRED PSNR dB↑，位置权重高的对的尺）**：

| | h1 | h16 | **h28** | **both-OOD** |
|---|---|---|---|---|
| baseline_fr（纯FR） | 25.55 | 21.86 | 22.09 | 20.41 |
| structpos_fr_pw30（承重） | 26.86 | 22.57 | 22.41 | 21.30 |
| **structcv_fr_pw100（承重+运动学）** | 26.11 | 22.74 | **23.34** | **21.67** |

承重+运动学 **长程 +1.25dB、both-OOD +1.26dB 净超纯 FR**。（latent 192 维聚合尺看不出——被黑盒稀释。）

### 4.3 跨域：只在光滑动力学域成立

| 域 | 承重+运动学 vs 纯 FR |
|---|---|
| uniform (a=0) | ✅ pixel h28 +1.25dB |
| parabola (a=常重力) | ✅ latent both-OOD 0.313→0.262 |
| collision (冲量,a 非光滑) | ❌ 失败，纯 FR 最好 |

**collision 失败**因为 smooth accel MLP + 硬承重都扛不住冲量 → 需要 **form-free** 的约束（阶段⑥ consistency loss，用户验证中）。

---

## 5. 方法论警示（源：diagnostic_report）

- **latent cos / K=4 ρ 涨 ≠ world model 真变好**：3/3 域上 best-by-pred_loss 与 best-by-K4-ρ 结论相反 → 主指标必须用 **pred_loss / pixel**，代理指标会误导。
- **高 λ_probe 会让 probe 项主导 optimizer**（probe/pred 损失比 3-125×），并造成 encoder 输出 intrinsic dim 塌方 → deep-supervision 权重不能无脑调大。
- **init 加载要验证**：曾因 `init_from_ckpt` 静默丢 192/216 ViT 权重导致 encoder 近随机初始化、结论全错。现有 `_remap_old_vit_keys` + 加载守卫（`loaded=216`）。

---

## 6. 代码沉淀（le-wm，均已入库，默认不影响 baseline）

| 开关 | 作用 |
|---|---|
| `wm.free_rollout` (默认 **true**) | 自回归多步训练，无 teacher forcing ← 主升力 |
| `loss.pos_weight` | 加重物理 slot 使其承重 |
| `loss.structured.weight` | 固定编码：`emb[:,:P]=proprio` |
| `dynamics.enabled / learnable_accel / accel_reg` | 二阶动力学头（pos_dim 自动适配域：uniform=2/collision=4） |
| `loss.consistency.weight / accel_weight` | form-free 速度/加速度一致性（冲 collision） |
| `freeze_encoder` | 冻结编码器（实测在 pusht→phyworld 迁移下**有害**，别用） |

脚本：[run_structdyn.sh](../run_structdyn.sh)（参数化 GPU/NAME/DATA/DOM + env SW/DYN）、[run_pixel.sh](../run_pixel.sh)（decoder→pixel rollout）。产物：`/data1/likun-share/junjxu/runs/structdyn_eval/`。

---

## 7. 待续

1. **consistency loss** 冲 collision（用户 A/B 验证中）——form-free 是否优于 accel-MLP。
2. **PIWM extrinsic 化**：要让运动学全面吃红利，需低维物理态做承重主 latent + 固定形式动力学 + 编码对外观鲁棒（量化/增广），而非黑盒 JEPA 挂 slot。
3. **pixel 作为默认评测尺**：latent 聚合会稀释物理增益。

---

**一句话**：这轮最大的确定收益是 **free-rollout（已默认）**；运动学能锦上添花，但要 free-rollout + 承重 + 光滑域 + pixel 尺四条齐备，冲量域另需 form-free 约束。
