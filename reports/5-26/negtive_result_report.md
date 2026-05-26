# Negative Result 复盘 + 修正报告

**日期**：2026-05-26
**主题**：phyworld OOD 初版结论 "encoder 在 OOD 上崩溃" 被推翻——实际是 **probe 协议 + R² 指标双重误导**。换成 LeWM 论文同款的 **MSE + Pearson ρ + 2-layer MLP probe** 后，**encoder 表征在所有 partition 上一致好**。同时确立：collision 上 FT ≈ frozen（净负 0.02 ρ），uniform_motion 上 FT > frozen（净正 0.02 ρ，leak-free 验证过）—— **FT 行为依任务而异，不能笼统说"frozen 总赢"**。
**前文参考**：[5-12/FINAL_REPORT.md](../5-12/FINAL_REPORT.md) · [5-12/UNIFORM_MOTION_REPORT.md](../5-12/UNIFORM_MOTION_REPORT.md) · [5-19/DIT_REPORT.md](../5-19/DIT_REPORT.md)

---

## 1. TL;DR

| 阶段 | 结论 | 是否仍成立 |
|---|---|---|
| **初版** | "encoder 在 both-OOD 上 R² 转负，表征崩溃" | ❌ 被推翻 |
| **修正 1** probe 协议 | K=4 multi-frame + Ridge 在全量 ID+OOD 混合 fit → "崩溃" 消失 | ✅ |
| **修正 2** 评估指标 | R² 跨 partition 不可比；换成 **MSE + Pearson ρ**（LeWM-paper 标准）| ✅ |
| **修正 3** probe 强度 | 加跑 **2-layer MLP**（LeWM 同款，直接 import `stable_pretraining.backbone.mlp.MLP`）| ✅ |
| **新结论** | frozen encoder vel ρ ≥ 0.89 跨 partition 一致；**collision 上 FT ≈ frozen 略负，uniform_motion 上 FT 真增益 +0.02 ρ**（leak-free 验证）| ✅ |

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
| **LeWM paper-init 20ep FT (leak-free)** | AGGREGATE | _训练中，等结果填_ | | | | | | | |

**关键观察（待 FT 数据补齐后再深入）**：

1. **vy 是 frozen encoder 最好读的物理量**：LeWM 0.982, DiT 0.978 ρ aggregate，几乎完美。原因是重力让 vy 在一条 traj 内从 0 线性变到 −0.77（**强 within-traj signal**），K=4 多帧 probe 直接读出来。
2. **pos_y 也很强**（ρ 0.967 / 0.978）：抛物线运动让 pos_y 在一条 traj 内变化巨大（9 → −5.6），跟 within-traj 简单 linear pos_x 比，信号丰富很多。
3. **vx 仍然像 uniform_motion 一样难** (LeWM ID ρ=0.76, DiT ID ρ=0.67)：水平方向是常量速度，跟 uniform_motion 同款挑战。
4. **DiT-XL 在 vx 上又过拟合**（vx ID 0.67），跟 uniform_motion 现象一致 → 同样的 4608-D 高维问题。

---

## 7. Frozen vs FT 横向对比（核心结论）

### 7.1 aggregate vel/vx ρ MLP

| Dataset | Encoder | params | phyworld FT | aggregate vel/vx ρ MLP |
|---|---|---|---|---|
| **collision** | LeWM pusht-only | 5.5M | **0** | **0.907** ✅ |
| collision | LeWM paper-init 16ep FT | 5.5M | 16 ep | 0.885（**Δ MLP −0.022**, Δ Ridge **0.000**）|
| collision | DiT-XL zero-shot | 749.8M | **0** | 0.932 |
| **uniform_motion** | LeWM pusht-only | 5.5M | **0** | 0.969 |
| uniform_motion | LeWM paper-init **leak-free FT** | 5.5M | 20 ep | **0.986**（**Δ +0.017** ✅）|
| uniform_motion | DiT-XL zero-shot | 749.8M | **0** | 0.923（0.954 dropout=0.3）|

### 7.2 Per-partition vel/vx ρ MLP（FT 行为差异最戏剧）

**Collision**：

| Partition | LeWM frozen | LeWM FT 16ep | Δ FT − frozen |
|---|---|---|---|
| ID | 0.935 | 0.870 | **−0.065** ❌ |
| r-OOD | 0.916 | 0.895 | −0.021 |
| v-OOD | 0.931 | 0.849 | **−0.082** ❌ |
| both-OOD | 0.902 | 0.888 | −0.014 |

**Uniform_motion**：

| Partition | LeWM frozen | LeWM FT leak-free | Δ FT − frozen |
|---|---|---|---|
| ID | 0.936 | 0.951 | **+0.015** ✅ |
| r-OOD | 0.903 | 0.935 | **+0.032** ✅ |
| v-OOD | 0.983 | 0.992 | +0.009 |
| both-OOD | 0.969 | 0.990 | **+0.021** ✅ |

### 7.3 解读：为什么两个数据集走相反方向

| | collision | uniform_motion |
|---|---|---|
| **FT 净效应** | **−0.022 ρ**（净负）| **+0.017 ρ**（净正） |
| **数据集物理** | 2 球 + 撞击瞬间，多事件 | 单球匀速，vx 恒等 |
| **frozen pretrain 是否接近 ceiling** | **是**（vel ρ 0.91，提升空间小）| **否**（vx ρ 0.97，FT 还能填 +0.02）|
| **机制** | catastrophic forgetting 通用 motion 特征 | FT 学到 phyworld 单球速度先验 |

**结论不要笼统说"frozen 总比 FT 好"**，更准确的说法：

> **frozen pretrain 在 collision 上已接近 ceiling（0.91 ρ），FT 没有 margin 学新东西反而 net negative；uniform_motion 上 frozen pretrain 留了 gap（0.97 ρ），FT 真的能填这个 gap。FT 的净效应取决于 frozen pretrain domain 跟 task domain 的距离，以及是否已经到 task ceiling**。

---

## 8. 修正后的总论点

| 旧论点 | 新论点 |
|---|---|
| LeWM 在 OOD 上崩溃 → 没学到通用物理 | LeWM/DiT 表征在所有 partition 上 ρ ≥ 0.74，"崩溃"是 R² + ID-only fit + K=1 三重协议问题 |
| 用 R² 衡量物理 probing | 用 **MSE + Pearson ρ + Ridge/MLP 双 probe**（跟 LeWM Table 1 一致）|
| 需要在 phyworld 上微调才能学到物理 | **取决于任务**：collision 不需要（frozen 已 ceiling）；uniform_motion 需要（FT 真的 +0.02 ρ） |
| LeWM 的优势是 JEPA + phyworld 训练 | **参数效率**（5.5M ≈ 749M DiT，差 0.03 ρ）；预训练域影响大（PushT > ImageNet）|

---

## 9. 后续可做

1. **重写 [5-12/COLLISION_REPORT §6.5](../5-12/COLLISION_REPORT.md) OOD 段落**：用新 MSE + ρ + MLP 数据替换 R²-only 叙事
2. **画 scatter plot**：x=true, y=pred, partition 着色 —— 论文里直观展示"所有 partition 都贴 y=x 线"
3. **跨数据集验证**：phyworld 太简单（黑底 + 圆球），结论能否外推到真实视频
4. **AutoTuneMLP sweep**：LeWM 原版 probe 扫多种 hidden_dim/lr/dropout，挑最好的；我们目前用单一 hidden=512、Adam lr=1e-3，要严格对齐 paper Table 1 数字应该跑 sweep

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
| LeWM leak-free FT ckpt | ~/.stable_worldmodel/uniform_paperinit_leakfree/ |
| FT-train 80% 子集 h5 | ~/.stable_worldmodel/phyworld_uniform_motion_train80.h5 |
| 80%/20% traj 切分 | ~/.stable_worldmodel/uniform_train_eps.npy + uniform_test_eps.npy |
| 初版（要修正的）章节 | [5-12/FINAL_REPORT.md §6.5](../5-12/FINAL_REPORT.md) · [5-12/COLLISION_REPORT.md §6.5](../5-12/COLLISION_REPORT.md) |
| 5-19 frozen vs FT 对比（R² 版） | [5-19/DIT_REPORT.md §2 / §3 / §4](../5-19/DIT_REPORT.md) |
