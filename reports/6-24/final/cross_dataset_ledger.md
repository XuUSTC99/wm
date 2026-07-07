# 跨数据集方法总账（phyworld / physion / physion++）

**日期**：2026-07-07
**目的**：合并三个并行 session 的结论，给论文一张"哪个方法在哪个数据集 work / 不 work / 为什么"的完整版图。避免重复、对齐口径。

> 来源 session：
> - **a29d4510**（本会话）：phyworld 方法探索（free-rollout / num_preds / 增广 / 动力学 / PIWM）
> - **50905d5d**（FINAL_SUMMARY_ANALYSE）：probe/structpos 论文消融
> - **356c245c**：Physion / Physion++ 迁移 + 真实数据训练

---

## 0. 一张表看全

| 方法 | phyworld(合成,有proprio) | physion zero-shot迁移 | physion++直训(真实,有proprio) | 判断 |
|---|---|---|---|---|
| **free-rollout** | ✅ 主升力 | ✅ 迁移最好(0.603，仍<random) | ✅ 长程最好 | **唯一三处通用** |
| num_preds 按域调 | ✅ collision≈20 | — | 待测 | 合成域有效 |
| **外观增广 app0.5** | ✅ 最强(both-OOD 腰斩) | ❌ 迁移无用 | **❌ 有害(nMSE 反转、崩100倍)** | **合成专属、伤真实** |
| 尺度增广 | ✅ collision | — | 待测 | 合成域 |
| pos_weight 承重 | 🔸 边际(pixel) | **❌ 迁移最差(0.551<random)** | ❌ 略损 | 越承重迁移越差 |
| structured/probe | 🔸 部分(可解释) | ❌ 伤迁移 | ❌ 无帮助 | 需 proprio；伤真实迁移 |
| 二阶动力学(MLP/PIWM) | ❌ 否 | ❌ | ❌ | 架构问题，到处否 |
| consistency | ❌ 无收益 | ❌ | ❌ 略损 | 否 |

**三条铁律**：
1. **free-rollout 是唯一跨合成/真实都正向的方法**（修了 LeWM 的 teacher-forcing 缺陷）。
2. **增广是合成域专属**：phyworld 大赢，但**真实 Physion++ 上 nMSE 反转成有害**（app0.5，friction_collision 崩 100 倍）。强度/类型在真实数据上要重扫，别照搬 0.5。
3. **物理结构（structured/probe/dynamics/pos_weight）到处不 work、且越强迁移越差**——根子是它把 encoder 过拟合到合成外观/合成物理，伤通用视觉。

---

## 1. 三个关键"反转/陷阱"（方法论，写进论文很值）

1. **latent cos 骗人**：多处 cos 上"probe 长程最稳 / app05 长程+0.08"，换 **nMSE / pixel 全部反转**。**主指标必须 pred_loss / nMSE / pixel，永不单用 cos。**
2. **zero-shot 迁移天花板 = random 架构先验(0.607)**：phyworld 训练最多恢复到接近、超不过；任何额外结构/增广/欠训都让它偏离、更差。**要突破必须在真实数据上训。**
3. **增广不跨域**：合成域治 r/m-OOD 的外观增广，在真实数据上因为外观本身就真实、反而破坏了有用信息。

---

## 2. 对论文的启示（定位）

- **Headline 主张**：`free-rollout`（去 teacher-forcing）是 OOD + 长程 + 合成→真实迁移的决定性、且唯一通用的杠杆。
- **物理结构（probe/structpos/dynamics）**：定位成**受控消融**，支撑"**让物理编码承重才是关键、具体结构 loss 次要**"，而非 headline（会被增广/free-rollout 碾压）——**此框架由 50905d5d 主导，本会话不重复**。
- **增广**：定位成**合成域 OOD 增益 + 一个反面教训**（不跨域），methodological 贡献（cos 陷阱）。
- **未解的真缺口**：physion_collide **无 proprio** → 所有物理监督失效。唯一可能的物理创新 = **无标签物理结构**（见 [physics_paper_design.md] §3B）。这是相对 PIWM(需弱proprio) 的真 delta，但给定"物理到处伤"的证据，**高风险**，需先便宜验证。

---

## 3. 分工（避免打架）

- **50905d5d**：probe/structpos 2×2 消融 + "速度进承重 slot" 改进（phyworld）。
- **356c245c**：Physion/Physion++ 迁移 + 增广在真实数据的重扫。
- **a29d4510（我）**：phyworld 主结论(已定稿) + 本总账 + 高风险增量"无标签物理"的便宜验证（不碰前两者地盘）。
