# 6-24 综合终报：物理结构化 World Model 的完整结论

**日期**：2026-07-05
**范围**：把 6-24 目录下 5 份报告的重点汇总成一份。研究主线——**怎么让 le-wm 的 latent world model 在 OOD 和长程 rollout 上变好**，试过 probe、固定编码、二阶动力学、free-rollout、承重编码等一系列手段。

> 源报告（细节见各自文件）：
> - [structured_loss_report.md](../structured_loss_report.md) — 固定编码位置(structpos)初步
> - [probe_vs_structpos_summary.md](../probe_vs_structpos_summary.md) — probe vs 固定编码 对比
> - [diagnostic_report.md](../diagnostic_report.md) — deep-supervision 方法论诊断
> - [piwm_dynamics_conclusion.md](../piwm_dynamics_conclusion.md) — 二阶动力学 + PIWM 对照 + free-rollout 发现
> - [kinematics_exploration.md](../kinematics_exploration.md) — 怎么让运动学真正帮到 OOD/长程
> - [final/general_augmentation.md](general_augmentation.md) — **通用增广（phyworld+physion 通吃的最强 OOD 方法）**
> - [novelty trace](../../../../.aris/traces/novelty-check/2026-07-09_run01/free_rollout_lewm_novelty.md) — free-rollout 作为 LeWM 后训练方法的查新边界

---

## 0. TL;DR（六条核心结论）

0. **⭐ 数据增广 = 最强通用 OOD 杠杆**（2026-07-06，见 [general_augmentation.md](general_augmentation.md)）：**外观增广全域治 r/m-OOD 软肋（nMSE 腰斩）+ both-OOD 大降**（uniform −48%、parabola −63%、collision −24%），**超过所有物理结构方法且纯视频（physion 也能用）**。
   - **collision 上最好的配方是"尺度增广 + 长 rollout(num_preds=20)"一起用**，both-OOD 从基线 0.393 降到约 0.21（−45%），三个随机种子跑出来都一样，不是碰运气。
   - **注意：这些方法不能想当然地全叠加，组合起来有时反直觉。** 比如外观增广和长 rollout 各自单独用都有效，但一起用反而更差（互相打架）；而尺度增广单独用是有害的，可一旦配上长 rollout 就成了全场最好（互补）。所以要挑对搭配，不是把所有有用的堆一起。
1. **真正的主升力是 free-rollout 训练（去掉 teacher forcing，LeWM原文采用teacher forcing 训练）**，不是任何物理结构化手段。三域（uniform/collision/parabola）一致大幅提升，both-OOD nMSE 砍半以上。**已设为默认。**
2. **训练 rollout 长度 num_preds 要按动力学复杂度调**（2026-07-05 新增，见 [optimization_plan §3](optimization_plan.md)）：**collision 甜点≈20**，both-OOD 0.393→**0.294（−25%）**、长程 h28 cos 0.58→**0.70**、ID 3 倍；但 **uniform/parabola 保持 8**（调大反而有害）。冲量/多步域吃长 rollout，简单域不吃。**这是 collision 上首个大提升。**
3. **运动学方程本身不带来提升**——直接挂 2 维 slot + smooth accel MLP，三域净贡献≈0 甚至变差；collision 完全不吃运动学。**2026-07-07 补充：严格照 PIWM 做（固定物理形式 `a=g` + 只学一个重力参数，parabola 上这就是正确物理）也一样没用——both-OOD 比基线还差（parabola 0.313→0.372、uniform 0.131→0.206），甚至比会过拟合的自由 MLP 更差**。根因：PIWM 的物理红利来自它整套 extrinsic 架构（独立低维物理 latent + 分阶段训练 + decoder 强制依赖物理态），不是单靠方程；嫁接到"共享黑盒切 2 维 slot"的 JEPA 上，2 维被 190 维黑盒淹没、还和黑盒预测器打架 → 越弄越差。**物理/动力学这条线彻底否掉（无论自由 MLP 还是严格 PIWM）；要吃红利须整换 extrinsic 架构，且大概率仍超不过增广。**
4. **运动学唯一出现过正收益的情形是 marginal 的、且已被增广彻底超越**：只有四条同时满足才行——① free-rollout 打底 + ② pos_weight 加重位置 loss + ③ 只在光滑动力学域（uniform/parabola）+ ④ 用 pixel PSNR 量（latent 聚合会稀释）。满足时比纯 free-rollout 好一点点（uniform 像素长程 +1.25dB、parabola both-OOD 0.313→0.262）。**但增广轻松做到 parabola 0.115（远超 0.262），所以这条实践上无用，只有"物理表示可解释性"的价值。**
5. **方法论教训**：latent cos / K=4 ρ 涨 ≠ world model 变好；主指标要用 **pred_loss / pixel**，别信被稀释或被 probe 主导的代理指标。consistency loss 实测未见收益。

**当前各域最优配方（both-OOD nMSE，↓）**：
- **uniform** = `free-rollout(np8) + 外观增广0.5` → **0.068**（基线 0.131）
- **parabola** = `free-rollout(np8) + 外观增广0.5` → **0.115**（基线 0.313）
- **collision** = `free-rollout + num_preds20 + 尺度增广0.5` → **~0.17**（基线 0.393，三种子确认）
- **跨数据集通用（含 physion）** = `free-rollout + 外观增广`（纯视频、无需 proprio）

---

## 0b. 试过啥、成没成（一张表看全）

| 手段 | 效果 | 用不用 |
|---|---|---|
| **free-rollout 训练**（去 teacher forcing） | 主升力，三域 both-OOD 砍半+ | ✅ **已默认** |
| **num_preds 按域调**（复杂域调大） | collision 甜点≈20，both-OOD −25% | ✅ 用（collision） |
| **外观增广**（亮度/对比度，强度0.5） | 全域治 r/m-OOD 软肋、both-OOD 大降、跨数据集通用 | ✅ **用（通用）** |
| **尺度增广**（中心缩放，强度0.5） | 配 num_preds20 是 collision 最优 | ✅ 用（collision） |
| 时序增广（帧-stride，想治 v-OOD） | 证伪：只是泛泛 helper，没治好 v-OOD | ❌ 没成 |
| 承重编码 pos_weight | 让运动学在 pixel 尺出现**边际**正收益 | 🔸 边际、已被增广超越 |
| 运动学（自由 MLP accel） | 过拟合，三域净贡献≈0 甚至变差 | ❌ 否 |
| **运动学（严格 PIWM，固定物理+学重力）** | 也没用，比基线和自由 MLP 都差（架构不对） | ❌ 否 |
| consistency loss（form-free 速度约束） | 实测无收益 | ❌ 没成 |
| 固定编码 structpos（emb 前 2 维=位置） | 短程/ID pixel 好，长程/both-OOD 反而差 | 🔸 部分 |
| probe（额外 head 读物理量） | 状态可读、速度可解码，但没解决长程漂移 | 🔸 部分 |
| freeze_encoder | pusht→phyworld 迁移下有害 | ❌ 别用 |
| slot 放大（位置→位置+速度） | 诊断清楚（我们非承重、PIWM 是独立完整状态 latent），但增广已封顶、ROI 低 | 🔒 搁置 |

**核心剩余软肋**：collision **v-OOD ~0.39**，appearance/scale/temporal 三种增广都压不下（纯视频难模拟未见速度）。

---

## 1. 研究脉络（试了什么，按顺序）

| 阶段 | 手段 | 结果 |
|---|---|---|
| ① probe | probe head 读出物理量 | 保长程一致 + 速度可解码，但没解决长程漂移 |
| ② 固定编码 structpos | 强制 `emb[:,0:2]=位置` | 短程/ID 变好，长程不变、both-OOD 退化 |
| ③ 二阶动力学（自由 MLP / 严格 PIWM） | 位置 slot 走 `z+v+accel` | **均否**：自由 MLP 过拟合、严格 PIWM 更差（架构不对，非方程问题） |
| ④ **free-rollout** | 训练时喂自己的预测（无 teacher forcing） | **三域全部大幅提升 ← 主升力** |
| ⑤ 承重编码 pos_weight | pred_loss 里加重物理 slot | 让运动学在 pixel 尺出现边际正收益（已被增广超越） |
| ⑥ consistency loss | 约束预测速度=真实速度（form-free） | 实测无收益 |
| ⑦ **num_preds 按域调** | 复杂域训更长 rollout | collision 甜点≈20，首个大提升 |
| ⑧ **数据增广** ⭐ | 外观/尺度/时序 像素扰动 | **性能天花板**：外观通用、尺度治 collision、时序没成 |

---

## 2. probe vs 固定编码（阶段 ①②，源：structured_loss / probe_vs_structpos）

数值均取自 [structured_loss_report.md](../structured_loss_report.md) 同一 run 三臂同测（uniform）。

| 维度（指标） | baseline | probe | structpos | 谁赢 |
|---|:--:|:--:|:--:|---|
| 速度可解码 ρ（ID，↑） | 0.50 | **0.70** | 0.60 | probe（structpos 居中） |
| 短程 pixel PSNR（h=1，dB↑） | 26.49 | 26.90 | **28.17** | structpos（+1.68 vs base） |
| ID pixel PSNR（dB↑） | 24.37 | 24.71 | **25.73** | structpos（+1.36 vs base） |
| 长程 pixel PSNR（h=28，dB↑） | **20.64** | 19.93 | 19.49 | **baseline**（两者都倒挂） |
| both-OOD latent nMSE（↓） | **0.295** | 0.341 | 0.407 | **baseline**（两者都变差） |

> ⚠️ **别被 latent cos 骗**：latent cos 尺上 probe 长程"最稳"（h28 0.882 vs base 0.843）、both-OOD 也微高，但换到可信的 **pixel / nMSE 尺，probe 和 structpos 在长程/both-OOD 双双跌破 baseline**。这正是 §5 那条方法论教训（latent cos 涨 ≠ world model 变好）的活例。

**结论**：probe 赢在"状态可读"、structpos 赢在"短程/ID pixel"，但**两者都只约束"状态是什么"、没约束"状态怎么变"，长程与 both-OOD 上都没超过 baseline，没解决长程漂移**。这直接引出阶段③④。

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

| 开关 | 作用 | 评价 |
|---|---|---|
| `wm.free_rollout` (默认 **true**) | 自回归多步训练，无 teacher forcing | ✅ 主升力 |
| `wm.num_preds` | 训练 rollout 步数（复杂域调大，collision≈20） | ✅ 按域调 |
| `aug.appearance` | 亮度/对比度增广（强度 0.5） | ✅ 通用最强 |
| `aug.scale` | 中心缩放增广（grid_sample，纯视频安全） | ✅ collision |
| `aug.temporal` | 帧-stride 速度增广（需 num_steps×temporal） | ❌ 没成 |
| `loss.pos_weight` | 加重物理 slot 使其承重 | 🔸 边际 |
| `loss.structured.weight` | 固定编码：`emb[:,:P]=proprio` | 🔸 部分 |
| `dynamics.enabled / learnable_accel / accel_form / accel_reg` | 二阶动力学头（`accel_form`: mlp=自由 / const=严格 PIWM `a=g`；pos_dim 自动适配域） | ❌ 均否 |
| `loss.consistency.weight / accel_weight` | form-free 速度/加速度一致性 | ❌ 无收益 |
| `freeze_encoder` | 冻结编码器 | ❌ 有害，别用 |

脚本：[run_structdyn.sh](../run_structdyn.sh)（参数化 GPU/NAME/DATA/DOM + env SW/DYN）、[run_pixel.sh](../run_pixel.sh)（decoder→pixel rollout，已支持 collision/parabola 域）。产物：`/data1/likun-share/junjxu/runs/structdyn_eval/`。

---

## 7. 待续

1. **攻 collision v-OOD**（唯一剩余软肋 ~0.39）：纯视频增广都压不下，可能需数据层面（更宽速度分布训练数据）或非增广手段。
2. **把增广接到 physion 训练验证**（physionpp/physion_collide）——真正跑通"phyworld+physion 通吃"（注意 physion 迁移封顶，须在真实数据上训；偏 physion 会话地盘，需协调）。
3. **PIWM extrinsic 化**（若坚持物理路线）：需整套换成独立低维物理 latent + 分阶段 + decoder 强制依赖，而非黑盒挂 slot。但增广已封顶，ROI 低。
4. **collision decoder 待修**（欠训、分区标注异常），修好才能用 pixel 尺复核 collision。

---

**一句话**：这轮的性能天花板是 **数据增广**（外观通用、尺度治 collision，纯视频跨数据集），其次是 **free-rollout（主升力，已默认）+ num_preds 按域调**；**物理/运动学这条线（固定编码、二阶动力学、自由 MLP、严格 PIWM、consistency）实践上全部否掉**——它们的问题是架构（共享黑盒挂 slot），不是方程，而且都被增广甩开。

---

## 8. Novelty check：free-rollout 能不能作为论文创新点？

**结论**：不能把“把 LeWM 改成 free-rollout 训练”单独当成强 novelty。free-rollout / autoregressive rollout / multi-step rollout training 本身是序列模型、视频生成和 world model 里常见的 exposure-bias 修法。Reviewer 很容易说这只是把 teacher forcing 换成已有闭环训练协议。

更合理的 novelty 边界是：

| Claim | Novelty | 怎么写 |
|---|---:|---|
| LeWM 的 one-step / teacher-forcing 风格评估会掩盖长程漂移 | Medium | 作为诊断贡献：h=1 很好，但 h=16/28 和 OOD 明显崩 |
| 直接用 free-rollout 做 LeWM post-training | Low | 不单独主张方法创新，只作为必要训练协议 |
| free-rollout 暴露并优化物理 WM 的 long-horizon / OOD closed-loop dynamics | Medium | 可作为主线的一部分，但要配合物理漂移分析和三域证据 |
| free-rollout + physics-consistent latent dynamics / 物理一致性约束 | Medium-High | 更像论文方法：不是换训练技巧，而是为物理闭环演化设计目标 |

所以论文里不建议写：

> We propose free-rollout training for LeWM.

建议写成：

> We identify that one-step teacher-forced LeWM training masks compounding physical errors, and use free-rollout post-training as a closed-loop objective to expose and optimize long-horizon OOD dynamics. We further show that state alignment alone is insufficient; physical consistency must constrain latent temporal evolution.

中文主线：

> 我们发现 LeWM 的单步/teacher-forcing 式训练会掩盖闭环 rollout 中的误差累积；因此用 free-rollout 后训练直接优化长程 OOD 动力学。同时，probe/structpos/PIWM-style 实验表明，只约束“状态是什么”不够，必须约束 latent “怎么随时间演化”。

当前终报里的实验证据支撑这个写法：

- §2：probe/structpos 让状态更可读、短程更好，但 h=28 / both-OOD 反而不如 baseline；
- §3：free-rollout 三域都显著提升，是工程上的主升力；
- §4：承重+运动学只有边际收益，说明纯 slot/方程嫁接到共享黑盒 latent 不够；
- §0/§0b：最终最强结果来自 free-rollout + 按域 num_preds + 数据增广，而不是 “free-rollout alone” 或 “物理方程 alone”。

**最终定位**：free-rollout 是本文必要的训练范式和强 baseline，不是单独 novelty；真正可讲的贡献是“用 closed-loop/free-rollout 视角重新暴露 LeWM 的物理长程/OOD failure，并系统比较状态对齐、动力学约束和增广后，给出哪些机制真正有效”。

### 8.1 如果包装成“probe + struct slot + dynamics + free-rollout”的组合方法，够不够 AAAI？

**不建议这么包装。** 如果写成“probe 作为 loss 保长程一致性并让物理量可解码，固定物理量编码保证短程变好，二阶动力学增强可解释性，承重编码优化，free-rollout 进一步优化长程和 OOD”，novelty 反而会显得弱。

原因有两点：

1. **模块本身都不是强新东西**：probe / physical slot / 二阶 dynamics / free-rollout / loss reweighting 都有近邻工作，容易被 reviewer 看成 engineering stack。
2. **实验不支持每个模块都作为正贡献**：probe 和 structpos 主要改善可读性或短程；二阶动力学、严格 PIWM-style 和 consistency loss 基本被否；真正稳定提升来自 free-rollout、num_preds 按域调和增广。

按 AAAI 方法论文的标准，这种“五件套”风险很高：

| 写法 | Novelty | 风险 |
|---|---:|---|
| “我们提出 probe + slot + dynamics + rollout 的物理结构化 LeWM” | Low-Medium | 像拼已有模块，且 ablation 会暴露很多组件不增益 |
| “我们用 free-rollout 修 LeWM teacher forcing 问题” | Low | free-rollout 是常见 exposure-bias 修法 |
| “我们证明物理可解码 ≠ 物理可 rollout，并系统比较哪些结构真的有效” | Medium | 更像可靠诊断贡献 |
| “closed-loop training + 承重物理通道/物理一致性目标形成统一机制，并显著超过增广/FR baseline” | Medium-High | 需要更强实验，目前还不够 |

因此，不建议主张：

> We propose a unified framework combining probe loss, structured physical slots, second-order dynamics, load-bearing encoding, and free-rollout training.

更建议主张：

> Physical interpretability does not imply physical rollout consistency in latent world models.

对应论文贡献可以写成：

1. **诊断贡献**：LeWM-style one-step / teacher-forced 训练会掩盖 long-horizon 和 OOD drift。
2. **系统实证贡献**：probe、structpos、二阶 dynamics 分别改善可读性、短程预测或解释性，但都不能稳定解决长程/OOD。
3. **有效 recipe**：closed-loop free-rollout + domain-adaptive rollout length + augmentation 是当前最稳的改进；承重物理通道只在平滑动力学域提供边际收益。

这条线的 AAAI 风险更低，因为它不是声称“我发明了五个模块”，而是在回答一个更清楚的问题：

> 为什么 latent WM 里“能读出物理量”不等价于“能按物理规律稳定演化”？

**当前 novelty 评估**：

- 组合式 method novelty：约 **4/10**，不稳。
- 系统诊断 + 负结果 + 有效训练 recipe：约 **6/10**，有机会。
- 如果后续能把“承重物理通道 + closed-loop rollout objective”抽象成统一算法，并在 OOD/长程上显著超过 free-rollout + augmentation baseline：可到 **7/10** 左右。

对应 trace：`/home/likun-share/.aris/traces/novelty-check/2026-07-10_run01/composed_physics_lewm_aaai.md`。
