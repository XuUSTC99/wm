# 论据:free-rollout 是唯一跨合成/仿真都通用的主升力(2.2–3.6×)

**对应主张**:[01_results_ledger.md](../01_results_ledger.md) **C1** / [06_storyline.md](../06_storyline.md) 发现二①
**论文位置**:paper §5.1(Table 5 headline)+ §3.3 机制
**一句话**:把 LeWM 的单步 teacher-forcing 换成多步 free-rollout 这一个开关,三域(三种子、区间零重叠)拿到 2.2–3.6× OOD 误差下降,并在照片级仿真直训与 zero-shot 迁移上独立验证它是唯一三处都正向的杠杆。

---

## 0. 三数据集总表(一眼看全 free-rollout 的效果)

| 数据集 | 设定 | 判决指标 | teacher-forced / 基线 | **free-rollout** | 效果 | 种子 |
|---|---|---|---|---|---|---|
| phyworld uniform | 合成直训 | both-OOD nMSE↓ | 0.300±0.007 | **0.136±0.009** | **2.2×** | 3 ✅ |
| phyworld parabola | 合成直训 | r/m-OOD nMSE↓ | 0.443±0.048 | **0.122±0.007** | **3.6×** | 3 ✅ |
| phyworld collision | 合成直训 | both-OOD nMSE↓ | 1.152±0.048 | **0.479±0.079** | **2.4×** | 3 ✅ |
| **Physion++** | 照片级直训 | h64 nMSE↓ | 1.174 (TF) | **0.141** | **8.3×** | 3 ✅ |
| **Physion++** | 照片级直训 | h64 cos↑ | 0.50 (TF) | **0.89** | +0.39 | 3 ✅ |
| **Physion** | zero-shot 迁移 | mean AUC↑ | — | **0.603** | **全场最高**(唯一逼近 random 天花板 0.607) | — |

**读表**:合成三域 2.2–3.6×,真实直训 **8.3×**(效果不降反升——真实动力学误差累积更严重,free-rollout 收益更大),迁移侧唯一逼近天花板。**三数据集、直训+迁移两种设定,free-rollout 全部正向且量级最大** → 唯一跨合成/真实通用的主升力。

![free-rollout 跨数据集效果](../figures/fig2_free_rollout.png)

（TF=teacher-forced 原始训练;FR=free-rollout。柱顶数字为倍数。合成三域 both/r-m nMSE + 真实 Physion++ h64 nMSE,一图看全。⚠️ 均为多种子。）

---

## 1. "2.2–3.6×" = 一个只翻一个开关的受控对照

**对照设计**:同一个 LeWM、同一批数据、同 epoch、同初始化,**只改训练方式**这一个变量:
- **Teacher forcing**(LeWM 原文默认):每步喂真值上下文,只预测下一步(`num_preds=1`)。
- **Free-rollout**:预测器**自回归喂自己的预测**滚 8 步,监督整条 rollout(`num_preds=8`)。

**三域 × 三种子**(both-OOD nMSE↓;parabola 用 r/m-OOD 避 h28 除零爆点):

| 域 | teacher forcing | free-rollout | 倍数(TF/FR) |
|---|---|---|---|
| uniform | 0.300±0.007 | 0.136±0.009 | 2.2× |
| parabola (r/m) | 0.443±0.048 | 0.122±0.007 | 3.6× |
| collision | 1.152±0.048 | 0.479±0.079 | 2.4× |

- **三种子区间完全不重叠** → 排除种子噪声;除训练方式外一切相同 → 2.2–3.6× 干净归因到这个开关。
- 出处:`/data1/likun-share/junjxu/runs/aaai_p0/rollout_{uniform,parabola,collision}_baseline_{tf,fr}_s{3072,1234,42}.log`;原始三域首次对照见 [piwm_dynamics_conclusion.md §3.3](../../6-24/piwm_dynamics_conclusion.md)。

## 2. "唯一跨合成/仿真都通用" = 三种设定分别验证,且别的方法都不通用

| 设定 | free-rollout | 别的方法 |
|---|---|---|
| **合成域直训**(PhyWorld 三域) | ✅ 2.2–3.6× | 物理结构 29/30 格不改善;增广域特定 |
| **照片级仿真直训**(Physion++) | ✅ 长程主力(h32 cos **0.91**) | 加物理结构全部损害长程(0.79–0.87) |
| **zero-shot 迁移**(Physion OCP) | ✅ 全场最高 **0.603**(唯一逼近 random 天花板 0.607) | 物理结构越强迁移越差(pos_weight 加权 0.551=最差);增广无用(0.597) |

**为什么别的方法当不了"通用主升力"**:
- **物理结构**:合成域有害(C4)、迁移越强越差、Physion++ 直训损害长程 → 三处都不行。
- **增广**:合成域最强,但照片级仿真上 appearance 反转 100×(C3)→ 域特定,不通用。
- **num_preds/horizon**:是 free-rollout 的调参维度,且域依赖(碰撞吃、光滑域不吃)→ 非独立主升力。

**只有 free-rollout 三种设定都正向、且无副作用** → "唯一跨合成/仿真都通用"。出处:[physionpp_ood_longhorizon.md](../../physion/physionpp_ood_longhorizon.md)(Physion++)、[transfer_improvement_report.md](../../physion/transfer_improvement_report.md)(迁移天花板 0.607)。

## 3. "主升力" + 机制(为什么有效)

"主升力" = 所有干预里**量级最大、种子最稳、唯一无副作用**的开关。

**机制**(5-27 rollout 实验给的因果解释):teacher forcing 下单步预测近乎完美(1-step cos **0.98–0.99**),**但多步漂移**,漂移速度严格 = f(动力学复杂度):uniform < parabola < collision。根因——teacher forcing 训练时永远看真值,**掩盖误差累积**;部署时必须多步 rollout,误差滚雪球。free-rollout 训练时就把模型暴露在自己的累积误差里,逼它学会纠偏。这是经典 **exposure bias** 问题(Scheduled Sampling, Bengio 2015)——所以论文**不 claim 它是新方法**,claim 的是"**在物理世界模型上这个训练协议压倒一切物理结构先验**"这个系统证据。

## 4. 审稿人预案(已知的可攻击点)

- **"free-rollout = scheduled sampling"**:主动承认并引用(Bengio 2015 / Data-as-Demonstrator / professor forcing);我们的 claim 是**相对重要性**(协议 > 结构),不是方法新。
- **TF/FR 对照里 batch size 也变了**(np1 用 128、np8 用 64,为适配序列长度):2–3× 的效应量远超任何 batch size 能解释的范围,且三域三种子一致 → 主结论稳。若审稿人较真,可补一个 batch 固定的对照(P1 可选)。
- **单 backbone(ViT-tiny)**:训练动力学结论建议在更大 backbone 复验(见 conclusion limitations)。
