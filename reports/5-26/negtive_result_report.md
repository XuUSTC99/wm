# Negative Result 复盘 + 修正报告

**日期**：2026-05-26
**主题**：phyworld OOD 初版结论 "encoder 在 OOD 上崩溃" 被推翻——实际是 **probe 协议 + R² 指标双重误导**。换成 LeWM 论文同款的 **MSE + Pearson ρ + 2-layer MLP probe** 后，**encoder 表征在所有 partition 上一致好**。

**🔴 5-27 重大修订**：之前 §6.2 / §6.3 喊的 "leak-free" FT 仅是 **traj-level** leak-free（FT-train traj 不在 probe-test 里），实际 **partition-level 泄漏严重**（FT-train 含 84% OOD trajs）。改用 PhyWorld 官方 `*_30K.hdf5` 的 **ID-only 1000 trajs** 重做 LeWM + DiT FT 后发现：

- **LeWM FT** 真 ID→OOD zero-shot 增益约为 0（uniform Δ +0.003），唯一例外是 parabola r-OOD +0.050
- **DiT LoRA FT** 在 3 个域上**全部大幅净负**（collision Δ −0.065, uniform Δ −0.136, parabola Δ −0.055）
- **结论**：phyworld FT 几乎不带来 ID→OOD generalization。之前 leaked-FT 喊的 "+0.02 ρ" 大半是 OOD memorization。新数据写在 §6.4。
**前文参考**：[5-12/FINAL_REPORT.md](../5-12/FINAL_REPORT.md) · [5-12/UNIFORM_MOTION_REPORT.md](../5-12/UNIFORM_MOTION_REPORT.md) · [5-19/DIT_REPORT.md](../5-19/DIT_REPORT.md)

---

## 1. TL;DR

| 阶段 | 结论 | 是否仍成立 |
|---|---|---|
| **初版** | "encoder 在 both-OOD 上 R² 转负，表征崩溃" | ❌ 被推翻 |
| **修正 1** probe 协议 | K=4 multi-frame + Ridge 在全量 ID+OOD 混合 fit → "崩溃" 消失 | ✅ |
| **修正 2** 评估指标 | R² 跨 partition 不可比；换成 **MSE + Pearson ρ**（LeWM-paper 标准）| ✅ |
| **修正 3** probe 强度 | 加跑 **2-layer MLP**（LeWM 同款，直接 import `stable_pretraining.backbone.mlp.MLP`）| ✅ |
| **§6.1–6.3 结论（旧）** | frozen encoder vel ρ ≥ 0.74 跨 partition 一致；FT 净效应 = f(frozen-到-ceiling 距离)：collision Δ −0.022, uniform/parabola-vx Δ +0.02~+0.026, parabola-vy Δ +0.005 | ⚠️ 部分推翻——见下 |
| **§6.4 ID-only FT 结论（新, 5-27）** | 改用 PhyWorld 官方 ID-only 1k trajs 重 FT 后：**真正 zero-shot ID→OOD 增益接近 0 (LeWM) 或灾难性净负 (DiT)**。LeWM collision 净负、uniform vx 打平、parabola vx 只在 r-OOD 保留 +0.05；DiT LoRA 在 3 域全部大幅净负（最差 uniform vx Δ −0.136）。之前 §6.2/§6.3 喊的增益 **大半是 partition memorization** | ✅ |
| **Caveat** | 测的是 **state encoding**（从 emb 读当前帧 pos/vel），不是 future prediction。"能编码 vy" ≠ "能预测抛物线运动"——后者需要 ARPredictor + multi-step rollout，proposal 在 [arpredictor_rollout_proposal.md](arpredictor_rollout_proposal.md) | ⚠️ |

---

## 2. 初版 vs 修正：核心数据

`collision_eval.hdf5`：1635 trajs，4 partitions（ID/r-OOD/v-OOD/both-OOD，phyworld §3.1 定义）。

| Setup | both-OOD pos | both-OOD vel | both-OOD coll AUC |
|---|---|---|---|
| **初版** LeWM paper-init+8ep · K=1 · ID-only fit | R² = **−0.024 ❌** | R² = 0.218 | 0.678 |
| **修正** LeWM pusht-only · K=4 · mixed-fit · MSE+ρ + MLP | nMSE 0.32, **ρ +0.878** | nMSE 0.24, **ρ +0.902** | 0.950 |

初版的"OOD 崩溃"叙事完全是 **K=1 单帧 + Ridge 只见 ID + R² 跨 partition 不可比** 三个 bug 联合的假阴性。

---

## 3. 初版协议为何不对

1. **K=1 单帧 probe**：速度需要跨帧差分，单帧 emb 无法做减法 → vel R² 系统性偏低
2. **Ridge 仅在 ID 上 fit**：测的是"线性系数能否从 ID 区间外插到 OOD"，**不是 "encoder emb 里是否含 OOD 物理信息"**

修正协议（K=4 + 80% ID+OOD mixed fit）解决了这两件事。结果立刻显示 encoder 在 OOD 上表征 OK。

---

## 4. 评估指标：用 MSE + Pearson ρ（不用 R²）

### R² 的核心问题（简短）

R² = 1 − SSE/SST，**分母 SST = N × Var(y) 跟测试集自身方差强相关**。两个具体后果：
- **跨 partition 不可比**：窄 partition (Var 小) R² 看似差，宽 partition (Var 大) R² 看似好，**同样 RMSE 下 R² 差 0.3+**
- **容易负值化**：probe 外插到 OOD 时 R² 转负看起来像"encoder 崩溃"，实际是 SST 缩水 × bias 协同效应——**这是初版 negative result 的根源**

### 选用 MSE + Pearson ρ 的依据

| 度量 | 优点 | 用法 |
|---|---|---|
| **MSE** | 绝对预测误差，无 SST 除法 | 看 encoder 的"绝对精度" |
| **Pearson ρ** | bounded [−1,1]，**对 bias/scale 不敏感**，跨 partition 可比 | 看"线性可读性" |

**这跟 LeWM 论文 Table 1 完全一致**（MSE + Pearson r），跟 phyworld toy 不同度量协议，但跟 SSL 物理 probing 文献主流一致。

### MSE 量纲注意：raw vs normalized

**MSE 公式本身就是 `(1/N) Σ (y − ŷ)²`**，没有内置归一化。但 raw MSE 的数值跟**目标量纲**强相关，**跨数据集不直接可比**：

| Dataset | pos 范围 | pos std | raw MSE 期望量级 |
|---|---|---|---|
| phyworld collision pos | [−28, 35] | 5.5 | 数量级 1-10 |
| phyworld uniform_motion pos | [0, 21] | 4.0 | 数量级 0.1-1 |
| LeWM pusht pos | [−29, 580] | 104 | 数量级 100-1000 |

**为了让 MSE 数字跨数据集可比** + 跟 LeWM Table 1 量级对齐，我们的 §6 表格报的是 **normalized MSE**：

```
nMSE = (1/N) Σ ((y_i − ŷ_i) / σ_train)²       ← 还是标准 MSE 公式
     = (1/N) × (1/σ_train²) × Σ (y_i − ŷ_i)²
     = raw_MSE / Var(y_train)
```

等价说法：**对 z-score 归一化后的目标用标准 MSE**。MSE 公式没变，变的是用哪份目标数据（raw vs z-scored）。

**数学性质**：nMSE ≈ 1 − ρ² （在 Ridge probe + 大样本下精确等于）。

实测验证：

| Encoder | raw MSE | nMSE | ρ | 1 − ρ² |
|---|---|---|---|---|
| LeWM frozen collision pos | 5.34 | **0.195** | 0.899 | 0.192 |
| DiT-XL collision pos | 4.51 | **0.167** | 0.914 | 0.164 |

→ 跟 LeWM paper Table 1 量级一致。

后续表格 / 对比中 MSE 一律为 nMSE，直接从 §6 表读。

---

## 5. 2-layer MLP probe（LeWM 同款）

### 架构

直接 `from stable_pretraining.backbone.mlp import MLP`，调用 `MLP(D_in, [hidden, D_out], dropout=p)`：

```
Linear(D_in, 512) → ReLU → Dropout(p) → Linear(512, D_out) → Dropout(p)
```

**训练**：Adam(lr=1e-3), MSE loss, 50 epochs, batch 512, K=4 多帧 concat 输入（与 Ridge 同协议）。
**默认 dropout=0**（LeWM 默认）；**dropout=0.3** 作为 DiT-XL 高维特征过拟合的 ablation。

**对比意义**：Ridge = 线性可读上限；MLP = 非线性可读上限。差距 > 0 → encoder 有非线性编码的物理信号。

脚本：[probe_mlp_mse_pearson.py](../../phyworld/scripts/probe_mlp_mse_pearson.py)

### Ridge vs MLP aggregate ρ 对比

| Encoder | Domain | Ridge ρ | MLP ρ | Δ |
|---|---|---|---|---|
| LeWM pusht-only frozen | collision | 0.890 | **0.907** | +0.017 |
| LeWM pusht-only frozen | uniform_motion | 0.892 | **0.969** | **+0.077** 🔥 |
| DiT-XL zero-shot | collision | 0.914 | **0.932** | +0.018 |
| DiT-XL zero-shot | uniform_motion | 0.948 | 0.923 ⚠️ | −0.025（过拟合）|
| DiT-XL zero-shot dropout=0.3 | uniform_motion | — | **0.954** ✅ | 修过拟合 |
| LeWM FT leak-free 20ep | uniform_motion | 0.968 | **0.986** | +0.018 |
| LeWM FT collision 16ep | collision | 0.890 | **0.885** | −0.005 |

**关键发现**：

1. **collision MLP 增益小** (+0.01-0.02)：encoder 编的物理信号基本线性可读
2. **uniform_motion LeWM frozen MLP 增益巨大** (+0.077 aggregate, +0.13 ID, +0.16 r-OOD)：**encoder 编了非线性速度信号**，Ridge 漏掉，MLP 挖出来
3. **DiT-XL uniform_motion 高维 (4608-D × 512 hidden = 2.4M params) 默认过拟合**：dropout=0.3 修好

---

## 6. 完整数据表（K=4 mixed-fit, no projector, MLP dropout=0）

**MSE 列均为 normalized MSE** = `mean(((ŷ−y) / σ_train)²)` = raw MSE / Var(y_train)。等价于把目标 z-score 归一化后再用标准 MSE 公式算。数值跨数据集可比、跟 LeWM Table 1 量级一致；nMSE ≈ 1 − ρ²。

### 6.1 Collision（collision_eval.hdf5, 9483 test frames）

| Encoder | Partition | pos nMSE | pos ρ Ridge | pos ρ MLP | vel nMSE | vel ρ Ridge | vel ρ MLP | coll AUC |
|---|---|---|---|---|---|---|---|---|
| **LeWM pusht-only frozen** | AGGREGATE | 0.219 | +0.899 | +0.891 | 0.192 | +0.890 | **+0.907** | 0.953 |
| (5.5M, no phyworld) | ID | 0.024 | +0.943 | +0.967 | 0.066 | +0.898 | +0.935 | 0.976 |
| | r-OOD | 0.128 | +0.875 | +0.916 | 0.171 | +0.874 | +0.916 | 0.951 |
| | v-OOD | 0.036 | +0.929 | +0.961 | 0.068 | +0.849 | +0.931 | 0.964 |
| | both-OOD | 0.321 | +0.902 | +0.878 | 0.239 | +0.902 | +0.902 | 0.950 |
| **LeWM paper-init 16ep FT** | AGGREGATE | 0.214 | +0.892 | +0.893 | 0.217 | +0.890 | **+0.885** | 0.944 |
| (5.5M, FT on collision 16ep) | ID | 0.047 | +0.896 | +0.932 | 0.125 | +0.871 | +0.870 | 0.962 |
| | r-OOD | 0.143 | +0.865 | +0.906 | 0.188 | +0.879 | +0.895 | 0.936 |
| | v-OOD | 0.047 | +0.876 | +0.944 | 0.129 | +0.834 | +0.849 | 0.967 |
| | both-OOD | 0.299 | +0.902 | +0.886 | 0.258 | +0.904 | +0.888 | 0.941 |
| **DiT-XL zero-shot** | AGGREGATE | 0.110 | +0.914 | +0.944 | 0.135 | +0.914 | **+0.932** | 0.976 |
| (749.8M, no phyworld) | ID | 0.033 | +0.896 | +0.962 | 0.067 | +0.895 | +0.933 | 0.992 |
| | r-OOD | 0.128 | +0.893 | +0.916 | 0.179 | +0.896 | +0.898 | 0.975 |
| | v-OOD | 0.053 | +0.860 | +0.933 | 0.072 | +0.862 | +0.910 | 0.977 |
| | both-OOD | 0.121 | +0.923 | +0.953 | 0.132 | +0.926 | +0.948 | 0.975 |

### 6.2 Uniform_motion（phyworld_uniform_motion.h5, 6670 test frames）

| Encoder | Partition | pos nMSE | pos ρ Ridge | pos ρ MLP | vx nMSE | vx ρ Ridge | vx ρ MLP |
|---|---|---|---|---|---|---|---|
| **LeWM pusht-only frozen** | AGGREGATE | 0.035 | +0.978 | +0.981 | 0.060 | +0.892 | **+0.969** 🔥 |
| (5.5M, no phyworld) | ID | 0.009 | +0.985 | +0.990 | 0.040 | +0.803 | +0.936 |
| | r-OOD | 0.017 | +0.974 | +0.982 | 0.056 | +0.743 | +0.903 |
| | v-OOD | 0.044 | +0.980 | +0.983 | 0.045 | +0.938 | +0.983 |
| | both-OOD | 0.047 | +0.975 | +0.979 | 0.084 | +0.880 | +0.969 |
| **DiT-XL zero-shot** | AGGREGATE | 0.050 | +0.980 | +0.981 | 0.177 | +0.948 | 0.923 ⚠️ |
| (749.8M, no phyworld) | ID | 0.022 | +0.988 | +0.984 | 0.161 | +0.856 | 0.789 ⚠️ |
| | r-OOD | 0.032 | +0.979 | +0.980 | 0.136 | +0.820 | +0.808 |
| | v-OOD | 0.060 | +0.979 | +0.981 | 0.156 | +0.969 | +0.956 |
| | both-OOD | 0.063 | +0.980 | +0.980 | 0.227 | +0.956 | +0.927 |
| **DiT-XL dropout=0.3 ablation** | AGGREGATE | — | — | +0.982 | — | — | **+0.954** ✅ |
| | ID | — | — | +0.990 | — | — | +0.862 |
| | r-OOD | — | — | +0.985 | — | — | +0.840 |
| **LeWM paper-init 20ep FT (leak-free)** | AGGREGATE | 0.031 | +0.983 | +0.984 | 0.034 | **+0.968** | **+0.986** |
| (5.5M, FT 仅见 80% trajs) | ID | 0.006 | +0.989 | +0.994 | 0.028 | +0.923 | +0.951 |
| | r-OOD | 0.009 | +0.987 | +0.991 | 0.044 | +0.884 | +0.935 |
| | v-OOD | 0.043 | +0.983 | +0.984 | 0.032 | +0.979 | +0.992 |
| | both-OOD | 0.041 | +0.981 | +0.982 | 0.033 | +0.974 | +0.990 |

### 6.3 Parabola（phyworld_parabola.h5, 6119 test frames, 1056 trajs 来自 HF 官方 eval）

Parabola = 单球抛物线运动（水平 vx 常量 + 垂直 vy 受重力线性变化）。
**y 轴有重力 dynamics**：vy 在一条 traj 内从 0 线性变到 −0.77，pos_y 从 9 抛物线落到 −5.6。这跟 uniform_motion (vy ≡ 0) 完全不同——**y 维是 parabola 独特的物理信号**。

| Encoder | Partition | pos_x nMSE | pos_x ρ MLP | **pos_y nMSE** | **pos_y ρ MLP** | vx nMSE | vx ρ MLP | **vy nMSE** | **vy ρ MLP** |
|---|---|---|---|---|---|---|---|---|---|
| **LeWM pusht-only frozen** | AGGREGATE | 0.167 | +0.915 | 0.065 | **+0.967** 🔥 | 0.170 | +0.908 | 0.040 | **+0.982** 🔥 |
| (5.5M, no phyworld) | ID | 0.098 | +0.925 | 0.024 | +0.990 | 0.097 | +0.761 | 0.016 | +0.994 |
| | r-OOD | 0.101 | +0.919 | 0.037 | +0.986 | 0.129 | +0.741 | 0.015 | +0.993 |
| | v-OOD | 0.133 | +0.944 | 0.086 | +0.958 | 0.144 | +0.934 | 0.057 | +0.979 |
| | both-OOD | 0.234 | +0.899 | 0.080 | +0.960 | 0.226 | +0.912 | 0.050 | +0.979 |
| **DiT-XL zero-shot** | AGGREGATE | 0.057 | +0.972 | 0.048 | +0.978 | 0.299 | 0.864 ⚠️ | 0.085 | +0.978 |
| (749.8M, no phyworld) | ID | 0.019 | +0.977 | 0.011 | +0.997 | 0.180 | 0.672 ⚠️ | 0.057 | +0.989 |
| | r-OOD | 0.032 | +0.967 | 0.016 | +0.996 | 0.332 | 0.641 ⚠️ | 0.050 | +0.986 |
| | v-OOD | 0.083 | +0.966 | 0.075 | +0.963 | 0.196 | +0.920 | 0.122 | +0.974 |
| | both-OOD | 0.068 | +0.973 | 0.059 | +0.972 | 0.392 | +0.877 | 0.088 | +0.976 |
| **LeWM paper-init 20ep FT (leak-free)** | AGGREGATE | 0.149 | +0.922 | 0.061 | **+0.969** 🔥 | 0.124 | **+0.934** ✅ | 0.026 | **+0.987** 🔥 |
| (5.5M, 845-traj leak-free) | ID | 0.074 | +0.939 | 0.023 | +0.991 | 0.078 | **+0.825** ✅ | 0.010 | +0.996 |
| | r-OOD | 0.074 | +0.936 | 0.036 | +0.986 | 0.092 | **+0.823** ✅ | 0.011 | +0.995 |
| | v-OOD | 0.136 | +0.945 | 0.081 | +0.959 | 0.105 | +0.952 | 0.036 | +0.982 |
| | both-OOD | 0.210 | +0.908 | 0.074 | +0.963 | 0.163 | +0.937 | 0.031 | +0.985 |

**关键观察**：

1. **vy 是 frozen encoder 最好读的物理量**：LeWM 0.982, DiT 0.978 ρ aggregate，几乎完美。原因是重力让 vy 在一条 traj 内从 0 线性变到 −0.77（**强 within-traj signal**），K=4 多帧 probe 直接读出来。
2. **pos_y 也很强**（ρ 0.967 / 0.978）：抛物线运动让 pos_y 在一条 traj 内变化巨大（9 → −5.6），跟 within-traj 简单 linear pos_x 比，信号丰富很多。
3. **vx 仍然像 uniform_motion 一样难** (LeWM ID ρ=0.76, DiT ID ρ=0.67)：水平方向是常量速度，跟 uniform_motion 同款挑战。
4. **DiT-XL 在 vx 上又过拟合**（vx ID 0.67），跟 uniform_motion 现象一致 → 同样的 4608-D 高维问题。
5. **FT 显著拉高 vx（+0.064 r/agg ρ over frozen），vy 也微涨**。Per-partition vx ID: 0.761 → 0.825 (+0.064), r-OOD: 0.741 → 0.823 (+0.082)。FT 在 parabola 上同 uniform_motion **同向加分**——共同点：frozen 在常量速度方向有 gap，FT 让 encoder 把常量速度信号更线性化到 emb。
6. **FT 对 pos_y/vy（已经接近天花板）几乎无影响**：vy AGG 0.982→0.987（+0.005），pos_y 0.967→0.969（+0.002）。**ceiling 现象**，跟 collision FT 在 high-ρ 维度退化是不同的机制。

---

> **⚠️ Caveat: 这是 state encoding，不是 future prediction**
>
> 上表测的是 **probe 从 `encoder(frame_t)` 读出当前帧 pos/vel** 的能力（K=4 用 t−3…t 四帧 emb concat），完全没用 LeWM 的 ARPredictor 做 t+1 外推。**vy ρ=0.98 只说明 emb 里编码了当前 vy 的信息，不代表模型「理解了重力」或「能外推抛物线」**。
>
> K=4 做的多帧差分本质是数学恒等式（连续帧 pos 差 ≈ vel），不是物理律学习。要测「能预测抛物线运动」必须 (a) 训 ARPredictor + (b) 在多步 rollout 下评 MSE。目前结论上限：**encoder 把当前帧的瞬时 pos/vel 解出来了，包括 OOD 球大小/初速**。

---

### 6.4 ID-only FT 重做（**5-27 新数据，推翻 §6.2/§6.3 部分 FT 增益**）

#### 6.4.1 为什么重做

§6.2 / §6.3 喊的 "FT (leak-free)" 用的是 `phyworld_uniform_motion_train80.h5` / `phyworld_parabola_train80.h5`——这俩是 eval 文件按 traj-id 80/20 切的 80% 部分。问题：

- eval 文件本身就**混合了 4 个 partition**（ID + r-OOD + v-OOD + both-OOD）
- 按 traj-id 切 80% 后，FT-train 中 **84% 是 OOD trajs**（uniform: 141 ID / 922 total = 15%）
- 我之前 `[SANITY] probe test_eps ∩ FT train_eps = 0` 只验证了 **traj-level** leak-free，**没**验证 partition-level

这意味着 FT 训练时 encoder 已经"见过" r > 1.5 / v > 4 的球。"OOD partition ρ" 测的不是 zero-shot OOD generalization，是 **encoder + probe 都见过 OOD 80% trajs 后能不能泛化到 OOD 20%**。

#### 6.4.2 修正方案

用 PhyWorld 官方 HF `magicr/phyworld` 的 `id_ood_data/*_30K.hdf5` 数据：

| 文件 | 范围 | partition |
|---|---|---|
| `collision_30K.hdf5` (本地已有) | m=[0.5, 1.5], v=[1.0, 4.0] | 100% ID |
| `uniform_motion_30K.hdf5` (HF 下载) | r=[0.71, 1.39], v=[1.02, 3.97] | 100% ID |
| `parabola_30K.hdf5` (HF 下载) | r=[0.71, 1.39], v=[1.02, 3.97] | 100% ID |

转 LeWM h5 格式（chunks=(32,224,224,3)），各取 1000 trajs（≈ 32k 帧，跟 §6.2/§6.3 leaked-FT 数据规模相当）。LeWM FT 20 epoch（同 prior protocol），ckpt 保存到 `~/.stable_worldmodel/{collision,uniform,parabola}_paperinit_id1k/`。

Probe 仍在 eval h5（含全 4 partition）上做，K=4 mixed-fit MLP。

#### 6.4.3 Collision ID-only FT

| Encoder | partition | pos ρ MLP | **vel ρ MLP** | coll AUC |
|---|---|---|---|---|
| LeWM frozen | AGGREGATE | +0.891 | **+0.907** | 0.953 |
| | ID | +0.967 | +0.935 | 0.976 |
| | r-OOD | +0.916 | +0.916 | 0.951 |
| | v-OOD | +0.961 | +0.931 | 0.964 |
| | both-OOD | +0.878 | +0.902 | 0.950 |
| LeWM FT 16ep （**partition-LEAKED**, 旧 §6.1）| AGGREGATE | +0.893 | +0.885 | 0.944 |
| **LeWM FT 20ep ID-only**（新 5-27）| AGGREGATE | +0.886 | **+0.884** | 0.939 |
| | ID | +0.923 | +0.878 | 0.952 |
| | r-OOD | +0.891 | +0.880 | 0.931 |
| | v-OOD | +0.917 | +0.861 | 0.969 |
| | both-OOD | +0.884 | +0.892 | 0.936 |
| DiT-XL zero-shot | AGGREGATE | +0.944 | +0.932 | 0.976 |

**Δ frozen → ID-only FT (per partition vel ρ)**：ID **−0.057**, r-OOD **−0.036**, v-OOD **−0.070**, both-OOD −0.010 → **FT 全负**。
**Δ frozen → leaked FT 16ep**：ID −0.065, r-OOD −0.021, v-OOD −0.082, both-OOD −0.014 → 类似。

→ collision FT 无论 leaked 与否都伤 encoder。**catastrophic forgetting 主导**，没 ceiling 也救不回来。结论：**collision 上 FT 不可取**。

#### 6.4.4 Uniform_motion ID-only FT （vx，单球常量水平速度）

| Encoder | partition | pos_x ρ | pos_y ρ | **vx ρ** |
|---|---|---|---|---|
| LeWM frozen | AGGREGATE | +0.979 | +0.939 | **+0.971** |
| | ID | +0.984 | +0.989 | +0.931 |
| | r-OOD | +0.972 | +0.976 | +0.899 |
| | v-OOD | +0.981 | +0.909 | +0.983 |
| | both-OOD | +0.976 | +0.915 | +0.972 |
| LeWM FT 20ep （**partition-LEAKED**, 旧 §6.2）| AGGREGATE | +0.984 | +0.947 | **+0.986** |
| | ID | +0.994 | +0.992 | +0.955 |
| | r-OOD | +0.991 | +0.986 | +0.932 |
| | v-OOD | +0.984 | +0.912 | +0.992 |
| | both-OOD | +0.982 | +0.929 | +0.991 |
| **LeWM FT 20ep ID-only**（新 5-27）| AGGREGATE | +0.983 | +0.944 | **+0.974** |
| | ID | +0.993 | +0.991 | +0.938 |
| | r-OOD | +0.989 | +0.980 | **+0.887** ⬇ |
| | v-OOD | +0.983 | +0.910 | +0.980 |
| | both-OOD | +0.981 | +0.927 | +0.982 |
| DiT-XL zero-shot | AGGREGATE | +0.982 | +0.986 | +0.944 |

**Δ frozen → ID-only FT (vx ρ)**：ID +0.007, r-OOD **−0.012** ⬇, v-OOD −0.003, both-OOD +0.010 → **基本打平**。
**Δ frozen → leaked FT (vx ρ)**：ID +0.024, r-OOD +0.033, v-OOD +0.009, both-OOD +0.019 → **+0.02 ρ**

→ **之前 §6.2 喊的 "FT vx +0.017 ρ" 主要来自 partition leak**。去掉 OOD-traj-exposure 后 FT 在 r-OOD 上甚至**轻微退步**。

#### 6.4.5 Parabola ID-only FT

| Encoder | partition | pos_x ρ | pos_y ρ | **vx ρ** | **vy ρ** |
|---|---|---|---|---|---|
| LeWM frozen | AGGREGATE | +0.915 | +0.967 | +0.908 | +0.982 |
| | ID | +0.925 | +0.990 | +0.761 | +0.994 |
| | r-OOD | +0.919 | +0.986 | +0.741 | +0.993 |
| | v-OOD | +0.944 | +0.958 | +0.934 | +0.979 |
| | both-OOD | +0.899 | +0.960 | +0.912 | +0.979 |
| LeWM FT 20ep （**partition-LEAKED**, 旧 §6.3）| AGGREGATE | +0.922 | +0.969 | **+0.934** | +0.987 |
| | ID | +0.939 | +0.991 | **+0.825** | +0.996 |
| | r-OOD | +0.936 | +0.986 | **+0.823** | +0.995 |
| | v-OOD | +0.945 | +0.959 | +0.952 | +0.982 |
| | both-OOD | +0.908 | +0.963 | +0.937 | +0.985 |
| **LeWM FT 20ep ID-only**（新 5-27）| AGGREGATE | +0.921 | +0.968 | **+0.925** | +0.986 |
| | ID | +0.936 | +0.990 | +0.770 | +0.994 |
| | r-OOD | +0.930 | +0.986 | **+0.791** ✅ | +0.996 |
| | v-OOD | +0.945 | +0.958 | +0.950 | +0.981 |
| | both-OOD | +0.908 | +0.962 | +0.930 | +0.983 |
| DiT-XL zero-shot | AGGREGATE | +0.972 | +0.978 | +0.864 | +0.978 |

**Δ frozen → ID-only FT (vx ρ)**：ID +0.009, r-OOD **+0.050** ✅, v-OOD +0.016, both-OOD +0.018
**Δ frozen → leaked FT (vx ρ)**：ID +0.064, r-OOD +0.082, v-OOD +0.018, both-OOD +0.025

→ **parabola 上 ID-only FT 在 r-OOD 上保留了 +0.05 增益**（半径不变性 FT 可能真学到了），但 ID 上的 +0.064 增益**几乎完全消失**——说明那部分是 OOD memorization。

vy 全部 ceiling（0.98+），FT 三个版本无差别。

#### 6.4.6 跨 3 域 ID-only FT vs frozen 总表（vel/vx ρ MLP）

| Domain · Partition | frozen | ID-only FT | Δ |
|---|---|---|---|
| collision · AGG | 0.907 | 0.884 | **−0.023** ❌ |
| collision · ID | 0.935 | 0.878 | **−0.057** ❌ |
| collision · r-OOD | 0.916 | 0.880 | −0.036 ❌ |
| collision · v-OOD | 0.931 | 0.861 | **−0.070** ❌ |
| uniform · vx · ID | 0.931 | 0.938 | +0.007 |
| uniform · vx · r-OOD | 0.899 | 0.887 | −0.012 |
| uniform · vx · v-OOD | 0.983 | 0.980 | −0.003 |
| uniform · vx · both-OOD | 0.972 | 0.982 | +0.010 |
| parabola · vx · ID | 0.761 | 0.770 | +0.009 |
| parabola · vx · r-OOD | 0.741 | 0.791 | **+0.050** ✅ |
| parabola · vx · v-OOD | 0.934 | 0.950 | +0.016 |
| parabola · vx · both-OOD | 0.912 | 0.930 | +0.018 |
| parabola · vy · ALL | 0.98+ | 0.98+ | 0 (ceiling) |

#### 6.4.7 核心论点变化

| 旧论点（§6.2 / §6.3 / §7）| 新论点（§6.4） |
|---|---|
| FT 净效应 = f(frozen-到-ceiling 距离) | 这个说法**部分成立但夸大了**——leaked FT 看到 OOD 后能学 partition-specific 信号，伪装成"填 ceiling gap" |
| uniform_motion FT +0.017 ρ ✅ | **真增益约 +0.000，之前的 +0.017 主要是 partition memorization** |
| parabola vx FT +0.026 ρ ✅ | **真增益 +0.022（保留），但 ID +0.064 → +0.009（消失大半），r-OOD +0.082 → +0.050（保留 60%）** |
| collision FT 净负 −0.022 ρ | **仍成立**——leaked / ID-only FT 都净负 |

唯一**真正的 zero-shot ID→OOD 增益**：parabola r-OOD vx，+0.050 ρ。可能反映 encoder FT 时学到了半径不变性（ball size 跟 vx 解耦），所以 r-OOD（球更大但 v 在 ID 范围）上能保留些 FT 收益。

#### 6.4.8 DiT LoRA FT （ID-only）

**Setup**: DiT-XL-2-256 frozen base + LoRA rank=16 加在最后 4 个 transformer block 的 attn Q/K/V/out，AdamW lr=1e-4, grad_clip=1.0。同一 ID-only 1k h5 数据。

**训练稳定性**：collision 8ep ✅，parabola 8ep ✅，**uniform_motion 8ep 在 ep6 step899 NaN**（grad_clip 没救住，lora_norm 已涨到 7.8），改 4 epoch 重训成功。说明 DDPM LoRA FT on phyworld 即使加 grad_clip 仍接近发散边界。

**Collision DiT FT**:

| Encoder | partition | pos ρ | **vel ρ** | coll AUC |
|---|---|---|---|---|
| DiT-XL zero-shot | AGGREGATE | +0.944 | **+0.932** | 0.976 |
| | ID | +0.962 | +0.933 | 0.992 |
| | r-OOD | +0.916 | +0.898 | 0.975 |
| | v-OOD | +0.933 | +0.910 | 0.977 |
| | both-OOD | +0.953 | +0.948 | 0.975 |
| **DiT-XL LoRA FT 8ep ID-only** | AGGREGATE | +0.926 | **+0.867** | 0.925 |
| | ID | +0.933 | +0.831 | 0.959 |
| | r-OOD | +0.913 | +0.848 | 0.930 |
| | v-OOD | +0.870 | +0.782 | 0.893 |
| | both-OOD | +0.931 | +0.882 | 0.921 |

**Δ zero-shot → DiT FT (vel ρ)**: ID **−0.102** ❌, r-OOD −0.050, v-OOD **−0.128** ❌, both-OOD −0.066 → **DiT FT 在 collision 上灾难性净负**。

**Uniform_motion DiT FT** (vx):

| Encoder | partition | pos_x ρ | pos_y ρ | **vx ρ** |
|---|---|---|---|---|
| DiT-XL zero-shot | AGGREGATE | +0.982 | +0.986 | **+0.944** |
| | ID | +0.987 | +0.996 | +0.809 |
| | r-OOD | +0.983 | +0.995 | +0.766 |
| | v-OOD | +0.981 | +0.986 | +0.968 |
| | both-OOD | +0.981 | +0.977 | +0.958 |
| **DiT-XL LoRA FT 4ep ID-only** | AGGREGATE | +0.969 | +0.933 | **+0.808** |
| | ID | +0.959 | +0.965 | +0.543 |
| | r-OOD | +0.956 | +0.953 | +0.599 |
| | v-OOD | +0.973 | +0.924 | +0.844 |
| | both-OOD | +0.969 | +0.912 | +0.854 |

**Δ zero-shot → DiT FT (vx ρ)**: ID **−0.266** ❌❌, r-OOD **−0.167** ❌, v-OOD −0.124, both-OOD −0.104 → **更严重的灾难性遗忘**，ID vx 从 0.81 跌到 0.54。

**Parabola DiT FT**:

| Encoder | partition | pos_x ρ | pos_y ρ | **vx ρ** | **vy ρ** |
|---|---|---|---|---|---|
| DiT-XL zero-shot | AGGREGATE | +0.972 | +0.978 | +0.864 | +0.978 |
| | ID | +0.977 | +0.997 | +0.672 | +0.989 |
| | r-OOD | +0.967 | +0.996 | +0.641 | +0.986 |
| | v-OOD | +0.966 | +0.963 | +0.920 | +0.974 |
| | both-OOD | +0.973 | +0.972 | +0.877 | +0.976 |
| **DiT-XL LoRA FT 8ep ID-only** | AGGREGATE | +0.950 | +0.971 | **+0.809** | **+0.953** |
| | ID | +0.949 | +0.993 | +0.491 | +0.962 |
| | r-OOD | +0.942 | +0.990 | +0.648 | +0.963 |
| | v-OOD | +0.952 | +0.956 | +0.839 | +0.953 |
| | both-OOD | +0.947 | +0.963 | +0.841 | +0.948 |

**Δ zero-shot → DiT FT (vx ρ)**: ID **−0.181** ❌, r-OOD +0.007, v-OOD −0.081, both-OOD −0.036
**Δ vy ρ**: ID −0.027, r-OOD −0.023, v-OOD −0.021, both-OOD −0.028 → 连 ceiling 都掉了

#### 6.4.9 DiT vs LeWM ID-only FT 对比（per-partition Δ vel/vx ρ）

| | collision (vel) | uniform (vx) | parabola (vx) | parabola (vy) |
|---|---|---|---|---|
| **LeWM FT Δ avg** | −0.043 | −0.000 | +0.023 | +0.003 |
| **DiT FT Δ avg** | **−0.087** | **−0.165** | **−0.073** | −0.025 |

**核心观察**：

1. **DiT LoRA FT 比 LeWM JEPA FT 退化得更严重**，在所有 3 个域都净负，幅度是 LeWM 的 2-5 倍。
2. **DiT 的 catastrophic forgetting 主要伤 ID partition**（−0.10 到 −0.27），跟 LeWM 全 partition 缓慢退化不同。可能因为：
   - 749M 模型 + LoRA rank=16 表征容量极大，少量 phyworld 数据让 LoRA layer 偏移 ImageNet pretrain 的 attention 模式
   - DDPM noise prediction loss 跟 probe target (vel) 关系间接，FT 反而破坏了 zero-shot 已有的 vel 表征
   - LeWM JEPA loss 自带 history-aware 结构（predict next emb），更接近物理任务

3. **DiT zero-shot 仍是 collision/uniform_motion 上的最强 baseline**（即使 LeWM FT 后），FT 不应被推荐。LeWM frozen 是 parabola vy 的最强 baseline。

4. **Parabola r-OOD 的 +0.007 是 DiT FT 唯一非负的 partition**——跟 LeWM ID-only FT 在 r-OOD 上的 +0.050 同方向。两个架构都暗示 phyworld 数据可能教 encoder 学到了某种"半径解耦"，但 DiT 上幅度太小，可能是噪声。

---

## 7. Frozen vs FT 横向对比（核心结论）

> **5-27 修订**：原 §7.x 数据全部用的是 **partition-leaked FT**（FT-train 含 84% OOD trajs）。下表保留 leaked 数据但标清楚，新增 **ID-only FT** 列做对照（真正 zero-shot ID→OOD generalization 测量）。

### 7.1 aggregate vel/vx ρ MLP

| Dataset | Encoder | params | phyworld FT | aggregate vel/vx ρ MLP | 注 |
|---|---|---|---|---|---|
| **collision** | LeWM pusht-only | 5.5M | **0** | **0.907** ✅ | frozen baseline |
| collision | LeWM paper-init 16ep FT | 5.5M | 16 ep | 0.885（Δ **−0.022**）| partition leak |
| **collision** | **LeWM FT ID-only** | 5.5M | 20 ep | 0.884（Δ **−0.023**）| 真 zero-shot |
| collision | DiT-XL zero-shot | 749.8M | **0** | 0.932 | |
| **collision** | **DiT-XL LoRA FT ID-only** | 749.8M | 8 ep | 0.867（Δ **−0.065**）| 真 zero-shot, **大幅退化** |
| **uniform_motion** | LeWM pusht-only | 5.5M | **0** | 0.971 | frozen baseline |
| uniform_motion | LeWM paper-init "leak-free" FT | 5.5M | 20 ep | **0.986**（Δ +0.017）| partition leak |
| **uniform_motion** | **LeWM FT ID-only** | 5.5M | 20 ep | 0.974（Δ **+0.003**）| 真 zero-shot |
| uniform_motion | DiT-XL zero-shot | 749.8M | **0** | 0.944 | |
| **uniform_motion** | **DiT-XL LoRA FT ID-only** | 749.8M | 4 ep | 0.808（Δ **−0.136**）| 真 zero-shot, **灾难退化** ❌❌ |
| **parabola** vx | LeWM pusht-only | 5.5M | **0** | 0.908 | frozen baseline |
| parabola vx | LeWM paper-init "leak-free" FT | 5.5M | 20 ep | **0.934**（Δ +0.026）| partition leak |
| **parabola** vx | **LeWM FT ID-only** | 5.5M | 20 ep | 0.925（Δ **+0.017**）| 真 zero-shot |
| parabola vx | DiT-XL zero-shot | 749.8M | **0** | 0.864 | |
| **parabola** vx | **DiT-XL LoRA FT ID-only** | 749.8M | 8 ep | 0.809（Δ **−0.055**）| 真 zero-shot, 退化 |
| **parabola** vy | LeWM pusht-only | 5.5M | **0** | 0.982 | frozen baseline |
| parabola vy | LeWM paper-init "leak-free" FT | 5.5M | 20 ep | 0.987（Δ +0.005, ceiling）| partition leak |
| **parabola** vy | **LeWM FT ID-only** | 5.5M | 20 ep | 0.986（Δ +0.004, ceiling）| 真 zero-shot |
| parabola vy | DiT-XL zero-shot | 749.8M | **0** | 0.978 | |
| **parabola** vy | **DiT-XL LoRA FT ID-only** | 749.8M | 8 ep | 0.953（Δ **−0.025**, ceiling 也掉）| 真 zero-shot |

**核心观察**：
- aggregate ρ 上，leaked FT 的 "+0.017" 增益（uniform_motion）在 ID-only FT 上**几乎消失**，只剩 +0.003（噪声范围内）。Parabola vx 的 "+0.026" 保留约 65% (+0.017)。Collision LeWM FT 无论 leaked 与否都净负。
- **DiT LoRA FT 在 3 个域上全部大幅净负**，幅度是 LeWM FT 的 2-5 倍，最差 uniform vx 跌 0.136。DiT-XL + LoRA + 小规模 phyworld 数据是 "三重灾难" 配方。
- **DiT zero-shot 是 collision/uniform 上的最强 baseline**，FT 完全不应推荐。LeWM frozen 是 parabola vy 的最强 baseline。

### 7.2 Per-partition vel/vx ρ MLP — Leaked vs ID-only 对照

**Collision** (LeWM 5.5M)：

| Partition | frozen | FT 16ep leaked | Δ leaked | FT ID-only | Δ ID-only |
|---|---|---|---|---|---|
| ID | 0.935 | 0.870 | **−0.065** | 0.878 | **−0.057** |
| r-OOD | 0.916 | 0.895 | −0.021 | 0.880 | −0.036 |
| v-OOD | 0.931 | 0.849 | **−0.082** | 0.861 | **−0.070** |
| both-OOD | 0.902 | 0.888 | −0.014 | 0.892 | −0.010 |

→ 两版 FT 都净负，方向一致。Collision 是 catastrophic forgetting 主导。

**Uniform_motion vx**：

| Partition | frozen | FT 20ep leaked | Δ leaked | FT ID-only | Δ ID-only |
|---|---|---|---|---|---|
| ID | 0.931 | 0.955 | **+0.024** ✅ | 0.938 | +0.007 |
| r-OOD | 0.899 | 0.932 | **+0.033** ✅ | 0.887 | **−0.012** ⬇ |
| v-OOD | 0.983 | 0.992 | +0.009 | 0.980 | −0.003 |
| both-OOD | 0.972 | 0.991 | **+0.019** ✅ | 0.982 | +0.010 |

→ **去掉 partition leak 后 r-OOD 从 +0.033 变成 −0.012**——之前的 OOD 增益基本是 OOD memorization。

**Parabola vx**（水平方向常量速度）：

| Partition | frozen | FT 20ep leaked | Δ leaked | FT ID-only | Δ ID-only |
|---|---|---|---|---|---|
| ID | 0.761 | 0.825 | **+0.064** ✅ | 0.770 | +0.009 |
| r-OOD | 0.741 | 0.823 | **+0.082** ✅ | **0.791** | **+0.050** ✅ |
| v-OOD | 0.934 | 0.952 | +0.018 | 0.950 | +0.016 |
| both-OOD | 0.912 | 0.937 | +0.025 | 0.930 | +0.018 |

→ **ID 上的 +0.064 基本消失（+0.009），r-OOD 的 +0.082 保留 60%（+0.050）**。后者可能是 FT 真学到了半径不变性（ball size 跟 vx 解耦），所以即使 encoder 没见过 r-OOD 半径范围，FT 后的 emb 仍对半径变化更鲁棒。

**Parabola vy**（重力, ceiling）：3 版本全部 ~0.99，无差别。

### 7.3 解读：FT 净效应来自哪

| | collision | uniform_motion (vx) | parabola (vx) | parabola (vy) |
|---|---|---|---|---|
| **leaked FT Δ ρ** | −0.022 | +0.017 | +0.026 | +0.005 |
| **ID-only FT Δ ρ** | **−0.023** | **+0.003** | **+0.017** | +0.004 |
| **去 leak 后保留比例** | 100% (净负不变) | **18%** | **65%** | 80% (ceiling) |
| **数据集物理** | 2 球 + 撞击 | 单球匀速，vx 恒等 | 单球抛物，vx 恒等 | 单球抛物，vy 线性 |
| **机制（旧解释）** | catastrophic forgetting | "填 frozen pretrain 的 vx gap" | 同 uniform | ceiling |
| **机制（5-27 新解释）** | catastrophic forgetting，仍成立 | leaked 增益 = OOD memorization | leaked 增益部分=memo, 部分=半径不变性 | ceiling 不变 |

**修正后的结论**：

> **FT 净效应 = f(frozen-到-ceiling) 这个旧论点 50% 错**。真实情况是：
> - **真 ID→OOD zero-shot FT 增益接近 0**（uniform vx Δ +0.003, collision Δ −0.023）
> - **唯一真增益**：parabola r-OOD vx，+0.050——可能是 encoder 学到 "ball radius ⊥ vx" 这个 invariance
> - 之前喊的 "uniform/parabola 上 FT 净增益 +0.02 ρ"，约 **70-80% 来自 partition memorization**（encoder 在 FT 时见过 OOD trajs 学了 partition-specific 信号）
>
> 真正能从 frozen pretrain 不够的物理 prior 中获益的 case，目前只看到 parabola r-OOD 一例。collision 是 FT 反作用，uniform vx 是 FT 基本无效果。

---

## 8. 修正后的总论点

| 旧论点 | 新论点 |
|---|---|
| LeWM 在 OOD 上崩溃 → 没学到通用物理 | LeWM/DiT 表征在所有 partition 上 ρ ≥ 0.74，"崩溃"是 R² + ID-only fit + K=1 三重协议问题 |
| 用 R² 衡量物理 probing | 用 **MSE + Pearson ρ + Ridge/MLP 双 probe**（跟 LeWM Table 1 一致）|
| 需要在 phyworld 上微调才能学到物理 ID→OOD 泛化 | **5-27 修订：去 partition leak 后 FT 增益几乎全消失**——uniform vx FT Δ +0.003（噪声内），collision FT 净负，parabola vx 只在 r-OOD 上保留 +0.05（半径不变性）。frozen PushT pretrain 已经覆盖了大部分 ID→OOD generalization |
| FT 增益 = 填 frozen-到-ceiling gap | **错了 50%**：leaked FT 看似填了 gap，70-80% 增益是 partition memorization；ID-only FT 显示 gap 实际很难填 |
| LeWM 的优势是 JEPA + phyworld 训练 | **参数效率**（5.5M ≈ 749M DiT，差 0.03 ρ）；预训练域影响大（PushT > ImageNet）；**phyworld FT 本身基本不带来 ID→OOD 增益** |
| DiT LoRA 是个安全的 FT 选项 | **不是**——3 域全部灾难性净负（最差 −0.136 ρ），训练还容易 NaN（grad_clip 没救住 uniform 8ep）。不建议在小规模 phyworld 数据上做 DiT-XL LoRA FT |

---

## 9. 后续可做

1. **DiT LoRA ID-only FT × 3** ✅ 已完成，见 §6.4.8
2. **ARPredictor rollout 测试** ✅ 已完成，见 [rollout_results.md](rollout_results.md)。结论：ARPredictor 1-step 预测很准（cos 0.98-0.99 含 OOD），但多步 AR rollout 复合误差漂移（collision 最快、uniform 最慢），OOD 上漂移更甚。**能编码当前状态 ≠ 能长程预测轨迹**。proposal 原文 [arpredictor_rollout_proposal.md](arpredictor_rollout_proposal.md)
3. **重写 [5-12/COLLISION_REPORT §6.5](../5-12/COLLISION_REPORT.md) OOD 段落**：用新 MSE + ρ + MLP 数据替换 R²-only 叙事
4. **画 scatter plot**：x=true, y=pred, partition 着色 —— 论文里直观展示"所有 partition 都贴 y=x 线"
5. **AutoTuneMLP sweep**：LeWM 原版 probe 扫多种 hidden_dim/lr/dropout，挑最好的；我们目前用单一 hidden=512、Adam lr=1e-3，要严格对齐 paper Table 1 数字应该跑 sweep

---

## 10. 文件 / 数据索引

| 类别 | 路径 |
|---|---|
| Collision Ridge probe | [probe_ood_fullfit.py](../../phyworld/scripts/probe_ood_fullfit.py) |
| Uniform_motion Ridge probe | [probe_ood_uniform_fullfit.py](../../phyworld/scripts/probe_ood_uniform_fullfit.py) |
| **2-layer MLP probe（LeWM `MLP` 类）** | [probe_mlp_mse_pearson.py](../../phyworld/scripts/probe_mlp_mse_pearson.py) |
| MLP probe log（dropout=0）| artifacts/logs/mlp_probe_lewm_dropout0.log |
| MLP probe log（dropout=0.3 ablation）| artifacts/logs/mlp_probe_lewm_dropout03.log |
| MLP probe log（collision + FT）| artifacts/logs/mlp_probe_collision_with_ft.log |
| LeWM pusht-only collision emb | artifacts/embeddings/lewm_pusht_only_collision_eval_emb_52k_noproj.npy |
| LeWM 16ep FT collision emb | artifacts/embeddings/lewm_16ep_epoch16_collision_eval_emb_52k_noproj.npy |
| DiT-XL zero-shot collision emb | artifacts/embeddings/dit_xl_zeroshot_collision_eval_emb_52k.npy |
| LeWM pusht-only uniform_motion emb | artifacts/embeddings/lewm_pusht_only_uniform_motion_emb_37k_noproj.npy |
| DiT-XL zero-shot uniform_motion emb | artifacts/embeddings/dit_xl_zeroshot_uniform_motion_emb_37k.npy |
| **LeWM leak-free FT uniform_motion emb** | artifacts/embeddings/lewm_uniform_paperinit_leakfree_uniform_motion_emb_37k_noproj.npy |
| LeWM leak-free FT ckpt（uniform）| ~/.stable_worldmodel/uniform_paperinit_leakfree/ |
| FT-train 80% 子集 h5（uniform）| ~/.stable_worldmodel/phyworld_uniform_motion_train80.h5 |
| 80%/20% traj 切分（uniform）| ~/.stable_worldmodel/uniform_train_eps.npy + uniform_test_eps.npy |
| LeWM pusht-only parabola emb | artifacts/embeddings/lewm_pusht_only_parabola_emb_noproj.npy |
| DiT-XL zero-shot parabola emb | artifacts/embeddings/dit_xl_zeroshot_parabola_emb.npy |
| **LeWM leak-free FT parabola emb** | artifacts/embeddings/lewm_parabola_paperinit_leakfree_parabola_emb_noproj.npy |
| LeWM leak-free FT ckpt（parabola）| ~/.stable_worldmodel/parabola_paperinit_leakfree/ |
| FT-train 80% 子集 h5（parabola）| ~/.stable_worldmodel/phyworld_parabola_train80.h5 |
| 80%/20% traj 切分（parabola）| ~/.stable_worldmodel/parabola_train_eps.npy + parabola_test_eps.npy |
| Parabola encode + Ridge 脚本 | [encode_parabola_paperinit_leakfree.py](../../phyworld/scripts/encode_parabola_paperinit_leakfree.py) |
| MLP probe log（parabola + FT）| /tmp/mlp_probe_parabola_ft.log |
| Ridge probe log（parabola FT）| /tmp/lewm_parabola_ft_ridge.log |
| **🆕 ID-only FT（5-27 新数据）** | |
| PhyWorld 官方 ID 训练数据 (HF) | `magicr/phyworld/id_ood_data/{collision,uniform_motion,parabola}_30K.hdf5` |
| ID-only LeWM h5 (1000 trajs each) | ~/.stable_worldmodel/phyworld_{collision,uniform_motion,parabola}_id1k.h5 |
| LeWM data configs (id1k) | le-wm/config/train/data/phyworld_{collision,uniform_motion,parabola}_id1k.yaml |
| LeWM FT ckpt (ID-only, 3 域) | ~/.stable_worldmodel/{collision,uniform,parabola}_paperinit_id1k/ |
| LeWM ID-only FT emb | artifacts/embeddings/lewm_{collision,uniform,parabola}_paperinit_id1k_*_emb_*_noproj.npy |
| Encode + Ridge 脚本 (3 域统一) | [encode_paperinit_id1k.py](../../phyworld/scripts/encode_paperinit_id1k.py) |
| MLP probe log (3 域 ID-only) | /tmp/mlp_probe_all_id1k.log |
| Ridge probe log (3 域 ID-only) | /tmp/lewm_{collision,uniform,parabola}_id1k_probe.log |
| Audit script (partition mix 检测) | inline in conversation jsonl |
| DiT LoRA ID-only FT 脚本 | [dit_lora_ft_3domains.py](../../phyworld/scripts/dit_lora_ft_3domains.py) (updated 5-27, grad_clip=1.0) |
| DiT LoRA FT emb (3 域) | artifacts/embeddings/dit_xl_lora_id1k_{collision_eval_emb_52k,uniform_motion_emb_37k,parabola_emb}.npy |
| DiT LoRA FT weights (3 域) | artifacts/embeddings/dit_xl_lora_id1k_{collision,uniform_motion,parabola}/ |
| DiT FT train logs | /tmp/dit_lora_id1k_{collision,uniform_4ep,parabola}.log |
| 完整 MLP probe log (incl. DiT id1k) | /tmp/mlp_probe_all_id1k_with_dit.log |
| **ARPredictor rollout proposal** | [arpredictor_rollout_proposal.md](arpredictor_rollout_proposal.md) |
| 初版（要修正的）章节 | [5-12/FINAL_REPORT.md §6.5](../5-12/FINAL_REPORT.md) · [5-12/COLLISION_REPORT.md §6.5](../5-12/COLLISION_REPORT.md) |
| 5-19 frozen vs FT 对比（R² 版） | [5-19/DIT_REPORT.md §2 / §3 / §4](../5-19/DIT_REPORT.md) |
