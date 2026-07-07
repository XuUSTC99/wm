# 物理结构化 World Model —— 论文创新点设计与验证

**日期**：2026-07-07
**目标**：把"二阶动力学 + 固定物理位置 slot"做成能在 **phyworld / physion / physion++ 三者都 work** 的论文创新点。

---

## 0. 硬约束（决定整个设计）

| 数据集 | proprio(物理标注) | 物理监督(structured/probe)可用? |
|---|---|---|
| phyworld | ✅ 有 | ✅ |
| physion++ | ✅ 有(3D pos/vel from pkl) | ✅ |
| **physion_collide** | ❌ **纯视频 MP4，无** | ❌ |

**→ 任何靠 proprio 监督的物理编码在 physion_collide 上死掉。** "三数据集都好"要求物理结构能**无标签**学出来。这是第一约束。

## 1. 已证伪（论文不能 claim）

物理 slot（structured/dynamics/自由MLP/严格PIWM）在当前"共享黑盒切2维slot"架构里**不提升、反而变差**（parabola const 0.313→0.372），被增广甩开。**根因是架构不是方程**：z_p(2维) 被 z_v(190维黑盒) 淹没，z_v 冗余编码了位置 → 预测靠 z_v(会漂)，修 z_p 没用。

## 2. 论文定位（诚实且能站住）

不写"物理 slot 打败增广"（假的），写：
> **一个物理可解释、从合成迁移到真实、且无需物理标签的世界模型**。性能由 free-rollout + 增广扛；物理结构提供**可解释性 + 跨域物理接地 + 零标签**（PIWM 还需弱 proprio 监督，我们做到零 → 覆盖 physion_collide，这是真 delta）。物理 slot 只需**不拖后腿地共存**，不需超过增广。

## 3. 核心设计：让物理"承重"且"无标签"

两个改动对应两个问题：

**(A) extrinsic 分离 + 解耦（治"物理拖后腿"）**
- 显式拆 latent = [z_p 低维物理 + z_v 视觉]。z_p 走固定二阶动力学（学参数），z_v 走黑盒。
- **关键：解耦 z_v 使其不编码位置**（否则 z_p 仍被淹没）。用**梯度反转对抗**：一个 adversary 从 z_v 预测 z_p，encoder 被训得让 z_v 对 z_p 不可预测 → 位置信息被逼进 z_p → z_p 承重。
- **这个解耦是无标签的**（用 z_p 自身，不用 proprio）。

**(B) 无标签物理先验（治"physion_collide 无 proprio"）——真正的创新点**
- z_p 不靠 proprio 监督，靠"必须按二阶动力学平滑演化"这个**结构约束自组织**（无标签）。
- 有 proprio 处(phyworld/physion++)：额外把 z_p 接地到真值（可选 grounding）。
- 无 proprio 处(physion_collide)：只靠动力学先验 + 解耦。

## 4. 验证计划（命门优先）

| # | 实验 | 判据 | 状态 |
|---|---|---|---|
| **1(命门)** | 物理做成承重(pos_weight/extrinsic)后，phyworld 上还拖后腿吗 | 物理 ≥ 纯 free-rollout(不能负) | 进行中 |
| 2 | 物理 + 增广能共存吗 | physics+aug ≥ aug | 进行中 |
| 3 | 解耦对抗：z_v 去位置后 z_p 承重、物理生效吗 | both-OOD 降 | 待建 |
| 4 | **无标签物理先验**（去掉 proprio grounding）能自组织物理 latent 吗 | z_p 可解码位置 + 不掉性能 | 待建 |
| 5 | 三数据集端到端 | phyworld/physion/physion++ 都不掉、physion 可解释 | 最终 |

**命门逻辑**：实验 1 若"物理即使承重仍拖后腿"→ 物理连共存都做不到，论文物理只能当 interpretability 小节；若通过 → 上解耦+无标签，创新点成立。

## 5. 结果与定论（2026-07-07）

**命门 + 无标签验证全部失败。物理这条线诚实地死了。**

| 域 | 纯 free-rollout | label-free 物理 | grounded 物理(有proprio) |
|---|---|---|---|
| uniform | 0.131 | 0.171 ⚠️ | 0.166 ⚠️ |
| parabola | 0.313 | 0.359 ⚠️ | 0.392 ⚠️ |
| collision | 0.393 | 0.653 ⚠️ | — |

- **无标签物理全域掉**（期望最好是"持平"，结果是"掉"）→ §3B 的创新点赌注**失败**。
- **连有 proprio 的 grounded const（严格 PIWM 重力形式）也掉**（uniform 0.131→0.166）→ 即使完美标签 + 正确物理形式，物理 slot 仍拖后腿。

**物理全变体清单（全否）**：自由 MLP ❌ / 严格 PIWM const ❌ / 无标签 ❌ / grounded ❌ / consistency ❌ / pos_weight 承重 🔸(边际、被增广碾压)。

**根因（确认）**：架构，不是方程。共享黑盒切 slot → 物理约束和黑盒预测器打架。唯一理论出路 = 整换 extrinsic（独立物理 latent + decoder 重建 + 分阶段），但连 grounded 都掉、ROI 极低，不做。

## 6. 对论文的最终建议

- **物理不能当性能创新点**（在此架构下，所有变体都伤性能，跨三数据集皆然）。
- **Headline = free-rollout**（修 LeWM teacher-forcing，唯一跨合成/真实通用）+ **方法论发现**（latent cos 陷阱、增广不跨域）。
- **物理（probe/structpos/dynamics）= 受控消融 / 诚实负结果**（支撑"承重是关键、结构 loss 次要"，由 50905d5d 主导）。
- physion_collide 无 proprio 的缺口：**无标签物理也填不上**（本验证证伪）→ 该数据集只能靠 free-rollout（纯视频）。
