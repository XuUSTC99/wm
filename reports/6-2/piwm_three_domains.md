# PIWM Deep-Supervision — 三域统一对比（parabola / uniform_motion / collision）

**日期**：2026-06-01
**一句话**：把 [5-26 的 PIWM deep-supervision 实验](../5-26/piwm_deepsup_results.md)从 parabola 推广到 uniform_motion + collision。**结论：parabola 上"多帧监督(mf4)是正解"的结论 NOT 泛化**——最佳 probe 粒度依物理类型 + partition 而变。

---

## 0. 数据完整性核查（relay 重启是否污染结果）

本批 uniform/collision 经历了一次混乱的接力重启（杀进程 / wipe 目录 / 一度双进程同写 ckpt）。已 double-check 排除污染：

| 检查 | 结果 |
|---|---|
| 每臂 `config.yaml` 的 probe 参数 | ✅ frames/target 全对，无错乱（mf4=frames4, 单帧=frames1）|
| probe_loss 收敛 | ✅ 全部正常下降（如 uniform mf4 0.185→0.077）|
| ckpt 时间线 | ✅ arm2/3 ckpt 均来自污染清除后的干净 run（17:14 之后）|
| 训练崩溃 | ✅ 0 |

→ 反常结论是**真实的**，非 artifact。

---

## 1. 设置（三域同口径）

- **训练**：ID-only 1000 trajs（PhyWorld 官方 `*_30K`），LeWM FT 20ep，加 deep-supervision probe loss
- **4 臂**：`baseline`(无probe) / `pos-only`(训练单帧, target=proprio) / `pos+vel`(训练单帧, target=[pos,vel]) / `mf4`(训练多帧 frames=4, target=[pos,vel])
- **velocity 监督列**：parabola/uniform = `action`(速度)，collision = `state`(速度，因 collision 的 action 是加速度)
- **评估**：rollout（ARPredictor 自回归）在全 OOD eval 集；**所有解码列用推理 K=4**（拼 4 个预测 latent 再解 Ridge）
- ⚠️ "单帧/多帧"指**训练时 probe 吃几帧**；推理一律 K=4

---

## 2. parabola（自由落体 + 重力）

### vx 解码 ρ（K=4，水平常量速度）
| partition | baseline | pos-only | pos+vel | **mf4** |
|---|---|---|---|---|
| ID | 0.304 | 0.585 | 0.612 | **0.702** |
| r-OOD | 0.612 | 0.662 | 0.480 | **0.643** |
| v-OOD | **0.649** | 0.450 | 0.483 | 0.518 |
| both-OOD | **0.696** | 0.515 | 0.612 | 0.659 |

### vy 解码 ρ（K=4，重力→within-traj 强信号）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.870 | **0.987** | 0.983 | **0.987** |
| r-OOD | 0.834 | **0.979** | 0.959 | 0.976 |
| v-OOD | 0.873 | **0.923** | 0.915 | 0.879 |
| both-OOD | 0.794 | **0.935** | 0.930 | 0.881 |

### latent cos by horizon
| h | baseline | pos-only | pos+vel | **mf4** |
|---|---|---|---|---|
| 4 | 0.864 | 0.917 | 0.867 | **0.925** |
| 8 | 0.814 | 0.877 | 0.846 | 0.875 |
| 16 | 0.535 | 0.633 | 0.648 | **0.702** |

**parabola 结论**：mf4 综合最佳——vx 把单帧的负迁移修回（ID 0.70 最高）、长程 cos 最佳（h=16 0.702）；vy 各臂都好。**"多帧是正解"在 parabola 成立。**

---

## 3. uniform_motion（单球匀速直线，vx 恒定，vy≡0）

> vy 列全 nan：匀速运动 vy 恒为 0，Pearson ρ 对常量无定义。只看 vx。

### vx 解码 ρ（K=4）
| partition | baseline | **pos-only** | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.440 | **0.804** | 0.596 | 0.642 |
| r-OOD | 0.564 | **0.737** | 0.700 | 0.661 |
| v-OOD | 0.782 | **0.944** | 0.924 | 0.942 |
| both-OOD | 0.893 | **0.928** | 0.902 | 0.886 |

### latent cos by horizon
| h | baseline | **pos-only** | pos+vel | mf4 |
|---|---|---|---|---|
| 4 | 0.964 | **0.968** | 0.928 | 0.922 |
| 8 | 0.912 | **0.932** | 0.857 | 0.812 |
| 16 | 0.845 | **0.865** | 0.754 | 0.706 |

**uniform 结论：跟 parabola 完全相反——`pos-only`(训练单帧) 全面最佳，mf4 反而最差。** 多帧监督(posvel/mf4)的 cos 甚至**低于 baseline**（h=16: mf4 0.706 vs base 0.845）。**多帧在这里有害。**

---

## 4. collision（双球碰撞，action=加速度，velocity 在 state）

### vx 解码 ρ（K=4，vx1）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.241 | 0.602 | 0.657 | **0.722** |
| r-OOD | 0.327 | **0.599** | 0.463 | 0.573 |
| v-OOD | 0.091 | **0.402** | 0.389 | 0.351 |
| both-OOD | 0.272 | **0.601** | 0.419 | 0.413 |

### vy 解码 ρ（K=4，vy1）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.168 | 0.460 | 0.573 | **0.727** |
| r-OOD | 0.366 | **0.593** | 0.485 | 0.591 |
| v-OOD | 0.464 | 0.490 | 0.293 | **0.514** |
| both-OOD | **0.683** | 0.594 | 0.521 | 0.514 |

**collision 结论**：baseline 极差（vx v-OOD 仅 0.09——碰撞是稀疏事件，纯位置差分读不出），**任何 probe 都大幅改善**；但**最佳臂依 partition 而变**：ID 上 mf4 最好，OOD 上 pos-only 最好。

---

## 5. 跨三域核心结论：多帧监督 **不普适**

| 域 | 物理特征 | 最佳臂 | vs parabola |
|---|---|---|---|
| **parabola** | vy 受重力(within-traj 变)+ vx 常量 | **mf4(多帧)** | 基准 |
| **uniform** | vx 常量,vy≡0 | **pos-only(单帧)** | ❌ 相反 |
| **collision** | 稀疏碰撞事件,分段常量 | ID→mf4, OOD→pos-only(混合) | ❌ 不一致 |

> **"训练多帧监督是 deep-supervision 的正解"这个 5-26 结论，只在 parabola 成立，不泛化到 uniform/collision。最佳 probe 监督粒度依物理类型 + partition 而变。**

### 机制假说（待进一步验证，非定论）

核心变量似乎是 **velocity 在轨迹内是否变化** + **多少个 position 分量带信号**：

1. **parabola**：vy 因重力在 traj 内线性变化 → 多帧窗口能捕捉这个变化 → mf4 受益；且单帧监督 2D 位置(x,y)时被大范围的 pos_y 主导、损伤 pos_x → 多帧修回 vx。
2. **uniform**：vx **常量** → 多帧窗口里 4 帧速度几乎一样，多帧监督**没有额外信息**，反而更重的 probe head(4×192→out)过度约束 emb → 单帧足够且最好，多帧连 cos 都拉低。
3. **collision**：velocity 分段常量+碰撞瞬间跳变(稀疏) → baseline 几乎读不出 → 任何 probe 都大补；多帧在跳变落入窗口时(ID)有利，外推到 OOD 则单帧泛化更稳。

一个统一视角:**关键是 encoder 能否产出干净的"逐帧 position"表征**——若能,K=4 推理时的帧间差分就能"免费"恢复速度。单帧 position 监督直接优化这个;只有当速度在 traj 内显著变化(parabola vy)时,多帧监督才带来单帧给不了的信息。

---

## 6. Caveats

- **uniform vy = nan**:vy 恒为 0,ρ 无定义,非 bug。
- **baseline 的几何外推**:baseline 无 probe,靠 K=4 位置差分读速度,这在任何速度下都成立——所以 baseline 在高速 OOD(尤其 uniform/parabola vx)是个**意外强的基线**,probe 不一定能超过(见 5-26 §5.4)。
- **collision vx baseline 例外**:collision 碰撞稀疏,baseline 位置差分也读不出(0.09)→ 这里 probe 才显出大幅价值。
- **未做**:λ_probe / frames sweep;per-frame vs per-window 监督的更细消融;统计显著性(单 seed)。

---

## 7. 文件 / 复现

| 项 | 路径 |
|---|---|
| 本报告 | reports/6-2/piwm_three_domains.md |
| 自动数据表(uniform/collision) | reports/6-2/piwm_uniform_collision_results.md |
| rollout 日志(12 个) | reports/6-2/logs/rollout_{parabola,uniform_motion,collision}_{baseline,posonly,posvel,mf4}.log |
| 训练编排 | reports/6-2/run_piwm.sh（初版）+ run_piwm_phase2.sh（接力，GPU0+GPU2）+ run_evals.sh（eval 补跑）|
| ckpt | ~/.stable_worldmodel/{uniform,collision}_piwm_{probe,posvel,mf4}_id1k/ + parabola 同名 |
| probe 实现 | le-wm/train.py（`loss.probe.{enabled,weight,target,frames}`）|
| parabola 原始分析 | [reports/5-26/piwm_deepsup_results.md](../5-26/piwm_deepsup_results.md) |

注：路径前缀 `agent_memory` 已重命名为 `am`；GPU 进程用相对 `.venv/bin/python` 调用以隐藏家目录。
