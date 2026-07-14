# PIWM 官方实现移植 phyworld —— 计划与结果

**日期**：2026-07-13（代码已写好，待 GPU 空档启动）
**目的**：回应审稿预案里 *"没有外部方法 baseline（如 PIWM 原实现）"* 的质疑（[../aaai_paper/02_story_and_novelty.md](../aaai_paper/02_story_and_novelty.md) §5、04_todo P2-4）。用**官方代码原样**在 phyworld 上端到端跑 PIWM（extrinsic-continuous 变体），和 LeWM 各臂在同一 eval 协议下对比。

## 1. 官方代码考察（已完成）

- repo：`PIWM/piwm-official/`（clone 自 [Trustworthy-Engineered-Autonomy-Lab/PIWM](https://github.com/Trustworthy-Engineered-Autonomy-Lab/PIWM)，arXiv:2412.12870 正主；另有 piwm-principles = 2503.02143）。
- 核心极小（~500 行）：**stage1** VAE（224×224→128 维，lr1e-4，kl1.0，50ep）→ **stage2** MLP extractor（frozen-VAE μ→物理态，[256,128,64]，lr1e-3，100ep；放出的代码就是**精确 state 监督** = 论文 δ=0 档，其 Fig3/4 显示 δ=0 最强 → 对 baseline 最有利，堵"故意削弱"质疑）→ **stage3** 已知方程 dynamics + 可学物理参数（repo 只有 DonkeyCar bicycle model）。
- phyworld 数据完美适配：pixels 已是 224×224 uint8，proprio=2 维位置。

## 2. 移植方案（代码已写：[../../../PIWM/phyworld_port/train_piwm.py](../../../PIWM/phyworld_port/train_piwm.py)）

原则：**架构/超参逐行照搬官方，只换数据管道**；dynamics 按其 BicycleDynamics 风格给各域写已知方程（fixed form + learnable phys params）：

- uniform：`p_{t+2} = p_{t+1} + s·(p_{t+1}−p_t)`，s=exp(log_s) 可学（摩擦系数，init 1）
- parabola：同上 + 可学重力向量 g
- collision：接触/冲量**超出 PIWM"光滑已知方程"假设**（其三个域 cartpole/lunar/donkey 也全是光滑的）——如实标注框架不覆盖，只跑 per-ball 匀速版做参考或跳过。
- stage3 训练按论文 Alg.1/Eq.10：从 encoder 提取的前两帧位置初始化，free-rollout 到轨迹末，MSE 对标签。

## 3. 跑什么

| run | 域 | 说明 |
|---|---|---|
| piwm_uniform_d0 | uniform | δ=0（精确监督，论文最强档） |
| piwm_parabola_d0 | parabola | 同上；检验 g 是否学到重力 |
| （可选）piwm_*_d0.05 | 两域 | δ=5% 弱监督档，忠实其"weak supervision"卖点 |
| （可选）piwm_collision_d0 | collision | 标注"框架假设外"参考 |

单域全流程估 ~1-1.5h/GPU（VAE 50ep 是大头）。

## 4. 评测与对比口径（待写 eval_piwm.py）

镜像 `phyworld/scripts/rollout_eval_id1k.py` 协议：同一 eval h5、同一 OOD 分区（pl_2col）、同 horizon 对齐（HS=3 上下文，预测帧 3..T-1）。PIWM 直接预测物理位置 → 指标：

- **位置 ρ**（per-dim Pearson，by partition）——与 LeWM 的 "decoded pos ρ from ROLLED-OUT latents" 直接可比（LeWM 侧 probe 在 eval 集 80/20 训，PIWM extractor 在 train 集训，口径差异如实标注、方向偏袒 LeWM）；
- 位置 nMSE by partition / by horizon。

对比对象：LeWM baseline_fr（PRED pos ρ：ID 0.896/0.958、r/m 0.916、v 0.917/0.819、both 0.922/0.823）及各物理臂。

## 5. 结果（2026-07-13）

### 5.1 学到的物理参数（证明移植忠实、物理真学到了）

| 域 | s（速度保持/摩擦） | g（常加速度/重力） | 对照真值 |
|---|---|---|---|
| uniform | **1.001** | [0.00, 0.00] | 匀速 s=1、无重力 ✅ |
| parabola | **0.986** | [0.004, **−0.028**] | 数据实测重力 a_y≈−0.025 ✅ 几乎吻合 |

即固定形式 + 只学物理参数的 PIWM dynamics 在两域都恢复了正确物理。（注：stage-3 用 free-rollout 拟合；试过 teacher-forced 但抽取器 ±0.29 位置噪声导致误差变量衰减，uniform s 被打成 0.32，故用 free-rollout。）

### 5.2 rolled-out 位置 ρ by partition（PIWM vs LeWM free-rollout）

ρ = 两个位置维均值。**PIWM ID/v-OOD 更强，但 r/m-OOD、both-OOD 崩，LeWM 稳。**

| 分区 | PIWM uniform | LeWM uniform | PIWM parabola | LeWM parabola |
|---|---|---|---|---|
| ID | **0.96** | 0.93 | **0.98** | 0.70 |
| **r/m-OOD** | 0.33 ⚠️ | **0.89** | 0.56 ⚠️ | **0.74** |
| v-OOD | **0.97** | 0.87 | **0.98** | 0.72 |
| **both-OOD** | 0.48 ⚠️ | **0.87** | 0.44 | **0.51** |

![PIWM vs LeWM](../figures/fig2_piwm_vs_lewm.png)

### 5.3 判读（回应"无外部 baseline" + 强化论文主张）

- **PIWM 干净物理 + 直接位置读出 → ID/v-OOD 上比 LeWM 还准**（其已知动力学按构造完美外推速度）。所以这**不是"PIWM 烂"**，是诚实的强 baseline。
- **但 r/m-OOD/both-OOD 崩**：VAE 编码器只见过 ID 球尺寸，遇到没见过的半径/质量就把位置编错，dynamics 救不了坏初值。LeWM free-rollout 对外观/尺寸偏移鲁棒得多。
- **落点**：物理结构买到的是"干净的 ID 物理外推"，**买不到 OOD 鲁棒**——崩的是**编码表示**不是方程。正是论文主张（decodable≠load-bearing 的编码器版）。**同时补上"无外部方法 baseline"短板（P2-4）。**
- ⚠️ 口径公平性：LeWM 侧 ρ 是 probe（在 eval 集 80/20 训）读 rolled-out latent，PIWM 是直接位置输出——协议不同，**偏袒 LeWM**（probe 见过 eval 分布）；即便如此 PIWM 在 ID/v 仍更高，OOD 崩不是协议造成的。position nMSE 因两者归一化基不同不直接可比，只用 ρ 对照。

## 6. 原始数据源（供后续拉数）

| 内容 | 路径 |
|---|---|
| PIWM 训练日志（含 s/g 收敛） | `/data1/likun-share/junjxu/runs/piwm_baseline/redyn_{uniform_motion,parabola}.log` |
| PIWM eval（ρ/nMSE by partition & horizon, JSON） | `/data1/likun-share/junjxu/runs/piwm_baseline/eval_{uniform_motion,parabola}_d0.json` |
| PIWM ckpt（vae/ext/dyn） | `/data1/likun-share/junjxu/runs/piwm_baseline/ckpts/{vae,ext,dyn}_{uniform_motion,parabola}_d0.pt` |
| LeWM 对照（PRED pos ρ） | `/data1/likun-share/junjxu/runs/aaai_p0/rollout_{uniform,parabola}_baseline_fr_s42.log` |
| 移植代码 | `PIWM/phyworld_port/train_piwm.py`、`eval_piwm.py`（官方码 `PIWM/piwm-official/`） |
| 图 | `reports/7-11/figures/fig2_piwm_vs_lewm.{png,pdf}`、脚本 `make_figures.py` |
