# 通用增广优化 —— phyworld + physion 通吃的 OOD 方法

**日期**：2026-07-06
**动机**：物理结构方法（structured/consistency/pos_weight）只在有 proprio 处可用，且 **physion 迁移封顶、物理结构越加越差**（见 [../physion/transfer_improvement_report.md]）。需要一个**纯视频、跨数据集通用**的 OOD 杠杆。**结论：数据增广（尤其外观增广）就是它。**

---

## 0. TL;DR

1. **外观增广 = 最强通用 OOD 杠杆**：全域治 r/m-OOD 软肋（nMSE 腰斩）、both-OOD 大降；**纯视频，phyworld / physion_collide / physionpp 都能用**（不需 proprio）；且**超过所有物理结构方法**（uniform aug05 both-OOD 0.068 < 最好物理配置 0.109）。
2. **尺度增广 = collision 专属加成**：单独有害，但 **scale + num_preds≈20 = collision 新最优（both-OOD 0.208，−47%）**；对简单域（uniform/parabola）有害。
3. **增广的交互高度非平凡**：appearance×np20 冲突、scale×np20 最佳叠加、appearance×scale 温和叠加。**不能盲目全叠。**

---

## 1. 为什么要纯视频增广（physion 约束）

| 方法 | phyworld | physion_collide(无proprio) | physionpp(有proprio) |
|---|:--:|:--:|:--:|
| free-rollout / num_preds | ✅ | ✅ | ✅ |
| **增广（外观/尺度）** | ✅ | ✅ | ✅ |
| structured / consistency / pos_weight | ✅ | ❌ | ✅ |

physion_collide 是纯视频（MP4 无物体位置），物理结构损失全部失效。**只有 free-rollout + num_preds + 增广三者跨数据集通用。** 增广还额外缩合成→真实的外观 gap（虽然平行 session 证明它不足以突破 physion 迁移墙，但对 phyworld OOD 是大杀器）。

---

## 1b. 三种增广分别是什么（先读这个再看结果）

**数据增广 = 训练时对输入帧做随机扰动，逼编码器"不管画面怎么变、物理状态不变"，从而对 OOD 更鲁棒。** 都在像素层面、不需要任何物理标注，所以三个数据集通用。每种攻一个 OOD 维度：

| 增广 | 对画面做什么 | 攻哪个 OOD | 实现 |
|---|---|---|---|
| **外观增广** appearance | 随机调**亮度/对比度**（每条轨迹一个随机值，整段一致）：`新帧 = 原帧 × 随机对比度 + 随机亮度` | **r/m-OOD**：半径/质量变→球的明暗/外观变；逼编码器别靠明暗判位置 | `aug.appearance`（jitter 标准差，如 0.5） |
| **尺度增广** scale | 随机**放大/缩小整帧**（球看起来变大/变小），中心缩放，用 grid_sample | **尺寸 OOD**：半径变=球大小变；外观增广模拟不了"变大变小"，这个直接模拟 | `aug.scale`（缩放抖动幅度，如 0.5） |
| **时序增广** temporal | 随机**隔帧取**（每隔 1 或 2 帧），球看起来动得更快=不同速度 | **v-OOD**：速度分布外；但❌**证伪**——只给 1x/2x 两档离散速度，学不到连续未见速度 | `aug.temporal`（帧-stride 倍数） |

> 一句话记忆：**外观治"看起来什么颜色/明暗"，尺度治"看起来多大"，时序想治"动得多快"（但没成）**。三者都不碰物理标注，纯像素——这是它们能跨 phyworld/physion 通用的原因。

---

## 2. 外观增广（brightness/contrast jitter，per-trial）

实现：[train.py](../../le-wm/train.py) `aug.appearance`。both-OOD / r/m-OOD nMSE↓：

| 域 | 基线 both | aug both | 基线 r/m | aug r/m |
|---|---|---|---|---|
| collision | 0.393 | 0.301 (aug0.3) | 0.183 | **0.110** |
| uniform | 0.131 | **0.068** (aug0.5) | 0.173 | **0.057** |
| parabola | 0.313 | **0.115** (aug0.5) | 0.127 | **0.065** |

- **r/m-OOD 软肋全域腰斩** —— 这正是要治的病（半径/质量变→外观变→位置编码被带偏；外观不变性直接解）。
- **both-OOD：uniform −48%、parabola −63%、collision −24%。**
- **uniform aug0.5(0.068) 超过之前所有物理配置（最好 structcv_fr_pw100=0.109）。**

---

## 3. 尺度增广（per-trial 中心缩放，几何）

实现：[train.py](../../le-wm/train.py) `aug.scale`（grid_sample，纯视频安全，结构化损失时自动跳过）。

| collision | r/m | both | h28cos |
|---|---|---|---|
| 基线 | 0.183 | 0.393 | 0.581 |
| scale0.3（单独） | 0.221 | 0.427 ⚠️ | 0.493 |
| **scale0.3 + np20** | **0.063** | **0.208** 🎯 | 0.715 |
| app0.3 + scale0.3 | 0.092 | 0.262 | 0.740 |

- **scale 单独有害**（collision 0.427>基线；uniform 0.225>0.131）——几何扰动太强，短 rollout 消化不了。
- **scale + np20 = collision 新最优 both-OOD 0.208（−47%）** + r/m 0.063（全场最低）。
- 直接治 size-OOD（半径），是外观增广（光度）补不了的维度。

---

## 4. 增广交互（反直觉，必须记住）

| 组合 | collision both-OOD | 判断 |
|---|---|---|
| np20 单独 | 0.294 | — |
| appearance 单独 | 0.301 | — |
| appearance × np20 | **0.472** | ❌ 冲突，别叠 |
| scale × np20 | **0.208** | ✅ 最佳叠加 |
| appearance × scale | 0.262 | ✅ 温和叠加 |

**不能盲目全叠**：appearance+长rollout 互相干扰；scale+长rollout 却互补。

---

## 5. 各域最优配方（当前）

| 域 | 最优 OOD 配方 | both-OOD | vs 基线 |
|---|---|---|---|
| uniform | free-rollout(np8) + **appearance aug 0.5** | 0.068 | −48% |
| parabola | free-rollout(np8) + **appearance aug 0.5** | 0.115 | −63% |
| **collision** | free-rollout + **np20 + scale aug 0.5** | **0.172** | **−56%** |
| **通用默认**（跨数据集，含 physion） | free-rollout + **appearance aug** | 全域治 r/m-OOD | — |

**增广强度甜点=0.5**（2026-07-06 精调）：appearance 0.7/1.0 在 uniform/parabola 反而变差（过增广）；scale 在 np20 上越强越好（0.3→0.5：collision both-OOD 0.208→**0.172**，−56%）。种子确认进行中（bee1zhunr）。

### 5b. 时序（速度）增广 —— v-OOD 假设证伪
`aug.temporal`（随机帧-stride=变速度，纯视频、时序键一致 stride）本想治 v-OOD，**证伪**：temporal2 的 v-OOD 0.556 **还不如 appearance 0.376**（stride 2x 只给 2 档离散速度，学不到连续 OOD 速度）。

**硬结论：v-OOD 是没啃下来的骨头** —— collision 新最优的 v-OOD 仍 0.388；**appearance/scale/temporal 三种增广都压不下 v-OOD**，collision both-OOD 现被 v-OOD 卡住（r/m 已治好 0.055）。纯视频增广难模拟未见速度，真限制。

---

## 6. 结论

- **要一个跨 phyworld+physion 通用的 OOD 提升 → 用外观增广**（纯视频、全域有效、超物理结构方法）。
- **collision 追求极致 → scale aug + num_preds≈20**（both-OOD 0.208）。
- **增广交互非平凡**，按上表配，别盲叠。
- 物理结构方法（structured 等）退居次要：只在有 proprio 且光滑动力学域小幅加成，且不迁移到 physion。

**已确认（2026-07-06）**：
- **scale+np20 三种子 = 0.208 / 0.198 / 0.244（均值 ~0.21）** → collision 最优坐实，非 fluke。
- **三重组合 app+scale+np20 = 0.253** → 比 scale+np20(0.208) 差，**再证 appearance×np20 冲突，赢家里别加 appearance**。
- **parabola scale = both 0.256 / r/m 0.232** → both 略好但 r/m 变差，远不如 parabola appearance(0.115/0.065)。**坐实简单域用 appearance、不用 scale。**

产物：`/data1/likun-share/junjxu/runs/structdyn_eval/{train_,rollout_}*aug*,*scale*`；开关 `aug.appearance` / `aug.scale`（[lewm.yaml](../../le-wm/config/train/lewm.yaml)）。
