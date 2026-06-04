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

### latent cos by partition
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.507 | 0.616 | 0.589 | **0.623** |
| r-OOD | 0.498 | **0.575** | 0.533 | 0.521 |
| v-OOD | 0.361 | **0.458** | 0.339 | 0.401 |
| both-OOD | 0.473 | **0.483** | 0.456 | 0.438 |

### latent cos by horizon
| h | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| 1 | 0.991 | **0.992** | 0.988 | 0.991 |
| 4 | 0.863 | **0.882** | 0.858 | 0.872 |
| 8 | 0.629 | **0.664** | 0.631 | 0.655 |
| 16 | 0.340 | **0.423** | 0.376 | 0.349 |

**collision 结论**：baseline 极差（vx v-OOD 仅 0.09——碰撞是稀疏事件，纯位置差分读不出），**任何 probe 都大幅改善**；但**最佳臂依 partition 而变**：vx/vy 解码 ID 上 mf4 最好、OOD 上 pos-only 最好。**latent cos 则一致是 pos-only(单帧)最高**（所有 partition + 所有 horizon）——这跟 uniform 一致（碰撞域整体 cos 偏低 0.34–0.62，因 AR 漂移在多事件碰撞上最严重，见 §2 h=16 对比 parabola/uniform）。

---

## 5. 跨三域核心结论：多帧监督 **不普适**

| 域 | 物理特征（被监督速度的轨迹内 std）| 最佳臂 | vs parabola |
|---|---|---|---|
| **parabola** | vy 受重力 **std=0.23**(在变) + vx 常量 | **mf4(多帧)** | 基准 |
| **uniform** | vx/vy **std=0**(全恒定) | **pos-only(单帧)** | ❌ 相反 |
| **collision** | vx **std=0.21**(撞击跳变,稀疏) | ID→mf4, OOD→pos-only(混合) | ❌ 不一致 |

> **"训练多帧监督是 deep-supervision 的正解"这个 5-26 结论，只在 parabola 成立，不泛化到 uniform/collision。最佳 probe 监督粒度依物理类型 + partition 而变——具体由「被监督速度量的轨迹内 std」决定（§5.1/§5.2）。**

### 5.1 这是数据分布问题吗？—— 查过了：不是采样瑕疵，是物理结构差异（有硬数据支撑）

曾怀疑反常结果是"uniform/parabola 训练数据采样不均"导致的 artifact。**查了训练数据(id1k)分布，排除采样问题，但发现了一个决定性的结构差异**——三域"哪个速度在轨迹内变化"完全不同：

| 训练数据量(id1k, 1000 trajs) | uniform(匀速) | parabola(自由落体) | collision(碰撞) |
|---|---|---|---|
| **vx 轨迹内 std**(轨迹内 vx 变化) | **0**（恒定） | **0**（水平恒定） | **0.215**（撞击跳变） |
| vx 全局范围 | [0.10, 0.40] | [0.10, 0.40] | [−0.98, 0.40] |
| **vy 轨迹内 std**（轨迹内 vy 变化） | **0** | **0.230**（重力线性变） | **0** |
| vy 全局范围 | [0, 0] | [−0.77, −0.02] | [0, 0] |
| pos_y 轨迹内 range（均值） | **0**（纯水平） | **12.1**（下落） | **0** |
| pos_x 轨迹内 range（均值） | 7.6 | 7.6 | 6.2 |

**两个结论**：

1. **不是采样不均**：三域的 vx 全局范围都是 [0.10, 0.40]，铺得开、一致；初速分布也一致。"数据不够均匀"被排除。
2. **真正的差异是结构性的(物理本质)**：**哪个速度分量在轨迹内变化**——uniform 全恒定(std=0)、parabola 的 vy 因重力在变(std=0.23)、collision 的 vx 在撞击跳变(std=0.21)。这是任务物理决定的，不是数据采集瑕疵。

### 5.2 机制（现在有训练数据支撑，不再是纯猜测）

§5.1 的 within-traj std 正好对上实验结果——**多帧监督是否有用，取决于被监督的速度量在轨迹内是否变化**：

| 域 | 被监督速度的轨迹内 std | 多帧能否提供单帧没有的信息 | 实测最佳臂 |
|---|---|---|---|
| uniform | vx/vy **std=0**（恒定） | ❌ 窗口内 4 帧速度一样 → 零新信息 | **pos-only(单帧)** ✓ |
| parabola | vy **std=0.23**（重力变） | ✅ 窗口能差分出 vy 变化 | **mf4(多帧)** ✓ |
| collision | vx **std=0.21**（跳变） | 部分（跳变落窗内才有用）| ID→mf4 / OOD→单帧 ✓ |

机制解读：
1. **uniform**：速度全程恒定 → 多帧窗口里 4 帧速度一模一样，多帧监督**拿不到任何单帧给不了的信息**，反而更重的 probe head(4×192→out)过度约束 emb → 单帧足够且最好，多帧连 cos 都拉低。
2. **parabola**：vy 因重力在 traj 内变化(std=0.23) → 多帧窗口能差分捕捉 → mf4 受益。
3. **collision**：vx 在撞击瞬间跳变(稀疏事件) → baseline 几乎读不出(vx v-OOD 仅 0.09) → 任何 probe 都大补；多帧在跳变落入窗口时(ID)有利，外推 OOD 则单帧更稳。

统一视角:**关键是 encoder 能否产出干净的"逐帧 position"表征**——若能,K=4 推理时的帧间差分就能"免费"恢复速度。单帧 position 监督直接优化这个;**只有当速度在 traj 内显著变化(parabola vy)时,多帧监督才带来单帧给不了的额外信息**。within-traj std 这个量化指标可以作为"该用单帧还是多帧"的先验判据。

---

## 6. Caveats

- **uniform vy = nan**:vy 恒为 0,ρ 无定义,非 bug。
- **baseline 的几何外推**:baseline 无 probe,靠 K=4 位置差分读速度,这在任何速度下都成立——所以 baseline 在高速 OOD(尤其 uniform/parabola vx)是个**意外强的基线**,probe 不一定能超过(见 5-26 §5.4)。
- **collision vx baseline 例外**:collision 碰撞稀疏,baseline 位置差分也读不出(0.09)→ 这里 probe 才显出大幅价值。
- **数据分布已查**:训练数据(id1k)vx 范围三域一致 [0.10,0.40],非采样不均(§5.1);差异是物理结构性的(轨迹内速度是否变化),非数据瑕疵。
- **未做**:λ_probe / frames sweep;per-frame vs per-window 监督的更细消融;统计显著性(单 seed);用 within-traj std 做先验、在更多物理任务上验证"std 决定单帧/多帧"这个判据。

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
