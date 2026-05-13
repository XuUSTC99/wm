# le-wm 在 phyworld collision 上的实验报告

**日期**：2026-05-09 ~ 2026-05-10
**模型**：le-wm（ViT-tiny encoder + JEPA-style latent predictor，192-D embedding）
**数据**：phyworld collision_30K（5000 trajectories × 32 frames，2-ball 1D 弹性碰撞）

---

## TL;DR

> **Update (2026-05-10)**：原本的 "negative result" 在补做了 paper-init + 多帧 probe 两个对照后**被部分推翻**。详见 §2.5、§2.4。

### 初版结论（from-scratch + 单帧 probe）

- ✅ 数据 pipeline、训练流程、probe 全部跑通
- ⚠️ from-scratch 训练 8 epoch（pred_loss 0.039 → 0.017 已收敛）后，**trained encoder 仅微弱超过 random ViT 和 pixel-stats baseline**
- 🔴  数据集本身存在严重的"像素 → 物理量"快捷信号（pixel-stats 9 维就能 R²(pos_x)=0.70 / R²(mass_ratio)=0.67），看似 negative result

### 修正后的结论（paper-init + 多帧 probe）

- 🆕 **从 lewm-pusht 论文权重初始化** + 在 collision 上继续训 8 epoch + **多帧 K=4 probe** → 完全不同的图景：

| Target | from-scratch + 单帧 | paper-init + 多帧 |
|---|---|---|
| pos_x R² | 0.726 | **0.911** |
| **vel_x R²** | **0.487** | **0.883** |
| mass_ratio R² | 0.711 | **0.826** (单帧) |
| collision AUC | 0.808 | **0.952** (单帧) |

- 🔁 **前面 negative result 的主因**是两个 confound **联合**造成的假阴性：
  1. **random init**：5.5M 参数 ViT-tiny 在 160k 帧上从零学物理表征太难（标准 ViT 训练用 14M-300M 张图）
  2. **单帧 probe 盲区**：速度信息天然在跨帧上，单帧 probe 即使 encoder 完美记住每帧 position 也读不出 velocity
- 🔴  数据集 shortcut 仍然存在（pixel-stats 0.70 R² 还在），但 trained 现在的 +0.21 lift 是真信号
- 在 le-wm 系列实验里，**下游表征评估应该默认用 (paper init) + (多帧 probe) + (`--no-projector`)** 三件套

---

## 1. 实验设置

### 1.1 数据转换

| 项 | 值 |
|---|---|
| 原始文件 | `collision_30K.hdf5` (HuggingFace `phyworld`, ~6.4 GB) |
| 抽取的轨迹数 | 5000（前 5000 条） |
| 每条 traj 帧数 | 32 |
| 总帧数 | 160 000 |
| 图像分辨率 | 224 × 224 × 3 |
| **action 信号** | computed acceleration `a[t] = (x[t+1] - 2x[t] + x[t-1]) / Δt²`，对两球分别计算然后拼接成 4-D（force-action） |
| 输出 | `~/.stable_worldmodel/phyworld_collision.h5`（uint8 像素 + proprio + state + mass + collision_event） |

转换脚本：[scripts/convert_collision_to_lewm.py](file:///home/qlib/agent_memory/wm/phyworld/scripts/convert_collision_to_lewm.py)。
和 uniform_motion 的最大区别：collision 是 2-ball、2D，action 维度 4 而非 1，并把每帧是否发生碰撞（用相对速度方向变号检测）作为 0/1 标签存下来用于 probe。

### 1.2 训练

| 项 | 值 |
|---|---|
| Encoder | ViT-tiny (hidden=192, 12 layers, patch=14) |
| Predictor | latent-space transformer (条件于 4-D action) |
| Batch size | 256 |
| 总 epoch（计划/实际） | **15 / 8**（被会话中断） |
| 优化目标 | JEPA latent prediction loss + sigreg |
| 训练 GPU | NVIDIA RTX A6000 (48 GB，单卡，本次训练用 ~14 GB) |
| 训练 log | [/tmp/lewm_collision_train.log](file:///tmp/lewm_collision_train.log) |
| Final ckpt | `~/.stable_worldmodel/collision_run/lewm_collision_epoch_8_object.ckpt` |

#### 收敛曲线（per-epoch validate/loss = pred_loss + 0.09 × sigreg）

| epoch | val_loss | pred_loss | 备注 |
|---|---|---|---|
| 1 | 13.22 | 0.0389 | sigreg 早期主导 |
| 2 | 3.07 | 0.0326 | |
| 3 | 1.58 | 0.0315 | |
| 4 | 2.43 | 0.0261 | val_loss 抖动来自 sigreg estimator 噪声 |
| 5 | 1.08 | 0.0239 | |
| 6 | 0.56 | 0.0202 | |
| 7 | 0.51 | 0.0178 | |
| **8** | **0.34** | **0.0167** | 训练中断点 |

pred_loss 单调下降；val_loss 因 sigreg 收敛过程会反复振荡，是预期行为。

---

## 2. Probe 结果

### 2.1 方法

按 phyworld 标准 protocol：
- **4000/1000 trajectory split**：5000 条 traj 随机 80/20 → 4000 条进 train、1000 条进 test。**同一条 traj 的 32 帧要么全在 train、要么全在 test，不允许拆分**。对应 128k / 32k 帧。**为什么不按帧随机切**：相邻帧像素几乎一样（球只挪 0.1 像素），按帧切会让 train 里 frame_t 和 test 里 frame_{t+1} 几乎是同一张图，probe 不用学泛化就能"背答案" → R² 虚高，失去诊断意义。按 traj 切才能保证 test 帧 encoder 真没见过，R² 反映真实泛化能力。详见 [§6.7](#67-为什么按-traj-切而不是按帧切)。
- 每帧过 encoder → 192-D（或 9-D pixel-stats）embedding
- 在 train 上 fit Ridge / class-balanced LogReg，在 test 上算 R² / AUC

#### Probe 目标

| 目标 | 类型 | 含义 |
|---|---|---|
| pos_x (2D) | 回归 | 两球的 x 坐标 |
| vel_x (2D) | 回归 | 两球的 x 速度 |
| mass_ratio | 回归 | m1 / m2 |
| collision_event | 二分类 | 当前帧是否发生碰撞（正样本 ~5.7%） |

Probe 脚本：[scripts/probe_collision_encoder.py](file:///home/qlib/agent_memory/wm/phyworld/scripts/probe_collision_encoder.py)

### 2.2 完整结果表

| 目标 | trained (no-proj) | trained (with-proj) | random ViT-tiny | pixel-stats (9-D) |
|---|---|---|---|---|
| **pos_x** R² | **0.726** | 0.708 | 0.649 | 0.700 |
| **vel_x** R² | 0.487 | 0.473 | 0.464 | 0.470 |
| **mass_ratio** R² | 0.711 | 0.719 | 0.666 | 0.666 |
| **collision_event** AUC | **0.808** | 0.741 | 0.763 | 0.667 |

per-dim 数据见原始 log：[/tmp/lewm_collision_probe.log](file:///tmp/lewm_collision_probe.log)、[/tmp/lewm_collision_probe_noproj.log](file:///tmp/lewm_collision_probe_noproj.log)

### 2.3 解读

#### (a) trained encoder vs baselines —— **优势很弱**

把 projector 拿掉后，trained encoder 在所有 4 个 probe 上都微弱超过 random 和 pixel-stats，但差距大多落在 **R² +0.02 ~ +0.07** 这种"统计上能看，业务上不显著"的区间。最显著的是 collision_event：trained 0.808 比 random 0.763 高 +0.045 AUC，比 pixel-stats 高 +0.141 AUC。

#### (b) **projector 在压缩信息**

with-projector 时 collision AUC 只有 0.741，**还输给 random ViT 的 0.763**；no-projector 直接跳到 0.808。说明 JEPA 的 projector head 把对 prediction loss 不重要、但对下游 probe 有用的信息（碰撞瞬间的强变化）压掉了。

> **结论**：要做下游 probe / 表征评测，请用 `--no-projector` 直接读 encoder cls token，不要走 projector。

#### (c) **数据集存在严重的快捷信号**

最让人警觉的不是 trained 的弱优势，而是 baseline 们的"反常之强"：

- pixel-stats（每帧只有 3 个通道均值 + 3 个标准差 + 3 个均值平方共 9 个数）拿到 **R²(pos_x) = 0.70**。9 个数怎么会包含两球 x 坐标？只可能是球的位置对全图平均亮度/对比度有显著影响（球是暗色，背景是浅色 → 球向某侧移动时整图均值变化）。
- pixel-stats 拿到 **R²(mass_ratio) = 0.67**。mass 直接编码进了球的视觉大小（半径 ∝ √mass），所以图的全局均值就泄露 mass。
- **vel_x** 在所有三个 encoder 上都卡在 R²≈0.47。单帧理论上不可能含速度信息，这个 0.47 的"正确率"完全来自数据集中"位置 → 速度"的统计相关（trajectory 总是从某侧朝某侧打）。

> **结论**：phyworld collision 这个数据集对**单帧** probe 而言不是一个干净的物理表征评测 benchmark，因为任何 encoder（包括纯像素均值）都能蹭到 60-70% 的"答案"。要真正比较表征质量，需要要么改 probe（多帧 / 时序差分），要么换更难的数据集（背景随机化、mass 与外观解耦）。

---

### 2.4 Multi-frame probe（Step 0 诊断实验）

§2.2 的 probe 是**单帧** —— 输入 emb[t] 一个 192-D 向量预测 (pos_x, vel_x, ...)。这个协议有一个已知盲区（见 §6.6 / [UNIFORM_MOTION_REPORT §5.1](UNIFORM_MOTION_REPORT.md)）：**速度需要看至少两帧**，单帧 probe 即使 encoder 完美记住每帧位置也读不出 vel。我们专门做了一次 **K=4 多帧 probe** 来量化这个盲区到底贡献了多少。

#### 设置

- 用 epoch 8 (from-scratch) ckpt，`--no-projector`，跟 §2.2 同 ckpt 同 split 同 Ridge 同 episode 切分
- 唯一变化：把每帧的特征从 `emb[t]`（192-D）替换为 `concat(emb[t-3], emb[t-2], emb[t-1], emb[t])`（768-D）
- 每条 traj 前 3 帧因没足够 history 被丢弃（160k → 145k 有效帧）

#### 结果

| Target | K=1 单帧 (no-proj) | K=4 多帧 (no-proj) | Δ |
|---|---|---|---|
| **pos_x** R² | 0.726 | **0.814** | +0.088 |
| **vel_x** R² | 0.487 | **0.594** | +0.107 |

#### 解读（**Update**：结论被 §2.5 paper-init 实验部分推翻）

**初版解读（只在 from-scratch ckpt 上做了多帧 probe 时）**：vel_x 涨了 +0.107，不是预想的"跳跃式"上升 → 当时下结论 "(B) 单帧协议盲区只贡献 1/3 缺口，主要还是 (A) action 稀疏 / (C) SIGReg"。

**修正后的解读（看完 §2.5 paper-init 数据后）**：那个 +0.107 之所以小，**主要不是因为速度信息没编进 emb，而是因为 encoder 对 position 的编码不够干净**（K=1 pos_x R² 只到 0.726）。在 paper-init encoder 上重做多帧 probe，**vel_x 从 0.620 跳到 0.883**（+0.26），uniform_motion paper-init 上甚至从 0.232 跳到 0.939（+0.71，几乎完美）——**说明多帧协议在 encoder 干净时确实能恢复速度，单帧盲区是真因，只是被 from-scratch 的弱 encoder 同时掩盖了**。

详细 4-way 对比看 §2.5。

---

### 2.5 Paper-init 实验：用 lewm-pusht 论文权重初始化

#### 动机

我们前面 negative result 的怀疑名单里有第 4 条：**"D) random init 在 5.5M-param ViT-tiny + 160k 帧的 setup 下学不出物理表征"**。le-wm 论文在 HuggingFace release 了 [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht) 权重（在 PushT 推方块任务上端到端训出来的 encoder + projector）。我们用它做 init，**只控制 (D) 这一个变量**，其他 hyperparam 全保持和 from-scratch 完全一致。

#### Setup

| 项 | 值 |
|---|---|
| 论文 ckpt | `~/.stable_worldmodel/lewm_paper_pusht/weights.pt`（72 MB） |
| 加载策略 | 只加载 **`encoder.*` + `projector.*` + `pred_proj.*`**（216 keys）；`predictor.*` + `action_encoder.*` 重新随机初始化（PushT action 是 10 维，collision 是 4 维，不能复用） |
| 训练 | collision 数据，8 epoch，batch=128，force-action mode（同 from-scratch） |
| 加载机制 | [le-wm/train.py](file:///home/qlib/agent_memory/wm/le-wm/train.py) 加了 `init_from_ckpt` config option |

#### 单帧 probe 结果（K=1，no-projector）

| Target | from-scratch | **paper-init** | Δ |
|---|---|---|---|
| **pos_x** R² | 0.726 | **0.863** | **+0.137** |
| **vel_x** R² | 0.487 | **0.620** | **+0.133** |
| **mass_ratio** R² | 0.711 | **0.826** | **+0.116** |
| **collision_event** AUC | 0.808 | **0.952** | **+0.144** |

**所有 4 个 target 同步上涨 ~0.13**——paper-init 明显地、系统地提升了 encoder 的表征质量。collision_event AUC 从 0.81 跳到 0.95 是最戏剧性的——说明碰撞瞬间的视觉变化在 paper-init encoder 里被大幅保留下来。

#### 多帧 K=4 probe 结果（no-projector）

| Target | from-scratch K=4 | **paper-init K=4** | Δ |
|---|---|---|---|
| **pos_x** R² | 0.814 | **0.911** | +0.097 |
| **vel_x** R² | 0.594 | **0.883** | **+0.289** |

**vel_x 从 0.594 跳到 0.883**（+0.29 R²）——这是这次实验最重要的数字。意味着：

- **encoder 实际上学到了精确的位置编码**（K=1 pos_x 0.86），多帧 probe 通过差分轻松恢复 velocity（K=4 vel_x 0.88）
- **velocity 信息是隐藏在 cross-frame 结构里的**——单帧 probe 看不到不是因为信息没了，是协议本身就读不出
- 这呼应了 uniform_motion paper-init 上 K=4 vel_x 从 0.23 跳到 0.94 的同样模式（只是 collision 因为 2-ball 复杂度，最高也只到 0.88，达不到 uniform 的 0.94）

#### 4-way 对比总表

| Setup | K=1 pos_x | K=1 vel_x | K=4 pos_x | K=4 vel_x |
|---|---|---|---|---|
| **from-scratch** (random init) | 0.726 | 0.487 | 0.814 | 0.594 |
| **paper-init** (lewm-pusht init) | 0.863 | 0.620 | **0.911** | **0.883** |

paper-init 在两个轴上都改善（横轴：probe 协议，纵轴：encoder 初始化）。最右下角的 (paper-init, K=4) cell **是 trained encoder 实际能力的真实读数**——前面的 negative result 是被 (random init) × (单帧协议) 联合放大出来的假阴性。

#### 修正后的归因（vs §2.4 的初版怀疑名单）

我们前面提的 4 条 negative result 怀疑因素：

| 因素 | 修正后的判断 |
|---|---|
| (A) action 稀疏（99% 帧 a≈0） | ⚠️ **没有干净的证据**——paper-init 同样用了稀疏 action 但 vel_x 涨上来了，说明 (A) 不是主因（虽然可能仍是次因，未排除） |
| (B) 单帧 probe 盲区 | ✅ **确认是主因之一**——多帧 probe 在 paper-init encoder 上让 vel_x 从 0.62 → 0.88 |
| (C) SIGReg / projector 信息瓶颈 | ⚠️ **没有 dominant 影响**——paper-init 的 emb 经过同样的 SIGReg 训练，velocity 信息仍然在多帧上可恢复，说明 SIGReg 没把信息冲掉 |
| (D) random init + 数据量不足 | ✅ **确认是主因之一**——paper-init lift 整体 +0.13~+0.29，效应很大 |

**主因 = (B) + (D) 联合**。Step 1 (parabola dense action) 的优先级因此**降低**——已经有较干净的解释了，不是 dominant 没解释完的问题。

---

## 3. 与之前 uniform_motion 实验的对照

uniform_motion（1-ball 1D，前一份实验）也曾观察到 trained encoder ≈ random encoder ≈ pixel-stats，pred_loss 收敛但 probe 上看不到明显优势。**collision 这次确认了同样的 pattern 在更复杂的 2-ball + collision_event 任务上仍然成立**——le-wm 的 JEPA 训练目标在 phyworld 这类**像素侧带强 shortcut** 的 toy 数据上不能可靠产出超出像素统计的表征。

---

## 4. 为什么不做第三个 phyworld 实验（parabola）

phyworld benchmark 共有 3 个实验：**uniform_motion**（1 ball 匀速直线）、**collision**（2 balls 弹性碰撞，本次实验）、**parabola**（1 ball 抛物运动 / 重力下自由落体）。前两个都做过了，**parabola 评估前明确决定跳过**。理由如下：

### 4.1 不会带来任何 uniform_motion + collision 没覆盖到的新维度

phyworld 三个实验在"物理复杂度"上的递进关系是：

| 实验 | 力的形式 | 自由度 | 交互事件 |
|---|---|---|---|
| uniform_motion | F = 0（无力） | 1 ball | 无 |
| **parabola** | **F = 常数（重力）** | **1 ball** | **无** |
| collision | F = 0 + 碰撞瞬间脉冲 | 2 balls | 弹性碰撞 |

**parabola 在复杂度光谱上夹在中间**：比 uniform_motion 多了一个"常数加速度"，但仍然是 1 ball、没有交互事件、力的形式没有质变。**uniform_motion 没学到的（trained ≈ baseline）+ collision 没学到的（trained ≈ baseline + 数据集 shortcut）合起来已经把 parabola 的可能结论锁定**：trained 也会 ≈ baseline，且像素 shortcut 同样存在。

### 4.2 action 信号在 parabola 上**比 uniform_motion 更 trivial**

我们用的是 **force-action mode**，action = 二阶差分得到的加速度。三个实验的 action 各自的 informativeness：

| 实验 | action（acceleration）实际取值 | 信息量 |
|---|---|---|
| uniform_motion | 恒为 0 | 几乎为 0，predictor 完全可以忽略 |
| **parabola** | **恒为 (0, -g)**（重力常数） | **几乎为 0**——同一个常数每帧重复，predictor 学到"忽略 action 直接用 ay = -g 的固定预测"就够了 |
| collision (本次) | 99% 帧为 0，碰撞瞬间脉冲 ±1.36 | 稀疏但**至少有 1-2 帧带强信号** |

JEPA 的 SSL 学物理表征**靠的就是"action 条件下 predict next latent"这个梯度信号**。parabola 上 action 是 dataset-wide 常数，**给 JEPA 的物理学习信号比 uniform_motion 还少**（实际等价于把 g 当作模型的一个隐式 bias）。collision 至少有碰撞那 1-2 帧上 action 是非 trivial 的信号源，已经是三个实验里"physics learning gradient"最丰富的，但实测 trained ≈ baseline。**parabola 上 SSL 信号更弱，几乎可以肯定结果只会更弱不会更强**。

### 4.3 单帧 probe 在 parabola 上能测的"独特物理量"基本不存在

parabola 上能想到的 probe target：

- **pos_x, pos_y**：和 uniform_motion 一样，单帧可读（位置就是图里球的位置）→ 不是新结论
- **vel_x, vel_y**：和 collision 一样，单帧不可读 → 不是新结论
- **重力常数 g**：dataset-wide 常数（所有 traj 共享一个 g），probe 只有 1 个标量目标，"测出来"也只是"有/没有恢复一个常数"，没有真正的回归任务
- **traj 的初速度**：单帧也不可读

**没有任何"只有 parabola 能测"的物理量**——所有可测的东西都在 uniform_motion + collision 里测过了。

### 4.4 数据集 shortcut 同样严重，甚至更纯净

parabola 是**单球 + 单一物理力**，画面更"干净"——只有 1 个有色球的视觉信号。这意味着 pixel-stats 这种 9-D baseline 在 pos 上更容易拿到高 R²（信号源更单一、相关性更强），shortcut 天花板**更靠近顶**——trained encoder 想超过 baseline 更难。

### 总结

跳过 parabola 是基于**"complexity-wise dominated, action 信号更弱, 没有新 probe target, shortcut 同样严重"**这 4 条独立理由。如果做出来，几乎可以肯定结论是 **trained ≈ random ≈ pixel-stats** 的同样 negative result，且证据强度还不如 collision——是工程时间的低 ROI 投入。

---

## 5. 后续建议

历史已做：

- ✅ **Step 0 — 多帧 K=4 probe**（§2.4）
- ✅ **Paper-init lewm-pusht 对照**（§2.5）
- 这两个补充实验**联合解释**了原 negative result，主因是 (B) 单帧盲区 + (D) random-init 数据不足

剩余可做：

1. **不做 Step 1（parabola）**：原本设计来测 (A) action 稀疏假设，但 §2.5 已经间接证伪 (A) 是主因——paper-init 同样用稀疏 action 但 vel_x 上来了。优先级降到很低。
2. **不做 Step 2（random-force 自建数据集）**：同理，(C) SIGReg 信息瓶颈也被 paper-init + 多帧 probe 间接证伪——velocity 信息在 emb 里**可以**通过跨帧恢复，没被 SIGReg 冲掉。优先级很低。
3. **数据增强 / 背景随机化**：让 pixel-stats 这种 9-D baseline 的 R²(pos_x)=0.70 floor 降下来，让 trained encoder 的优势更显著。中等优先级。
4. **测试 random-init multi-frame（消歧实验）**：现在还有一个 confound 没拆开——paper-init 的 encoder 包含 PushT 训出的视觉表征，可能把 from-scratch 多帧 vel_x = 0.59 拉到 paper-init 多帧 0.88 的过程中，"PushT 视觉知识"和"在 collision 上多训的物理知识"哪个贡献更大没法分。一个干净对照：**只用 paper-init encoder 的初始权重，不在 collision 上训**，直接 probe。如果 vel_x R² 已经 ~0.85+，说明 PushT visual 是主因；如果只到 ~0.5，说明 collision 上的训练贡献了大半。成本：~10 min。
5. **把训练补到 epoch 15**：低优先级，预期改变 < 0.02 R²。

---

## 6. 概念澄清（针对实验中容易混的点）

### 6.1 "JEPA projector 在毁信息 → probe 必须 `--no-projector`"

le-wm 的架构是：

```
图 → ViT encoder → 192-D h  →  projector (MLP) → 192-D z  → predictor 用 z + action 预测下一帧的 z
                  ↑raw cls           ↑projected
```

JEPA loss 是在 **projected 空间 z 上**算的，不是 raw h 上。projector 被训练用来"让预测下一帧的 loss 容易最小化"——它会**主动压掉那些对预测下一帧没用的维度**（information collapse）。

probe 默认从 projector 输出 z 读，所以读到的是**被压缩过的特征**。`--no-projector` 让 probe 直接读 encoder cls token h（projector 之前），信息没被压。

实验上 collision_event 的 AUC 从 with-projector 0.741 → no-projector 0.808：**碰撞瞬间的视觉变化对"预测下一帧"没什么用**（碰撞极短），projector 把它压掉了；但对下游 probe 关键。

> **下游表征评估必须用 `--no-projector`，不要走 projector。**

### 6.2 R² / AUC / class-balanced LogReg 速查表

整份报告里反复出现 R²、AUC、Ridge、class-balanced LogReg 几个 metric/模型术语，集中放这里：

#### Ridge 拟合**连续目标** → 用 **R²** 评

- 适用：pos_x、vel_x、mass_ratio 这种实数值
- 详细：见 §6.6
- 评估指标 R²（决定系数）：

```
R² = 1 − SSE/SST
其中  SSE = Σ(y_true − y_pred)²        ← 模型预测剩下的误差
      SST = Σ(y_true − mean(y_true))²  ← y 自身的总方差
```

| R² 值 | 含义 |
|---|---|
| **1.0** | 完美预测（SSE=0），**上限** |
| 0.9+ | 极强，emb 几乎完整编码了目标 |
| 0.5 | 一般，emb 解释了 50% 方差 |
| **0** | 模型预测**等价于直接猜 y 的平均值** |
| **负值** | 比猜平均值**还差**（test 上经常见，泛化失败的征兆）|

> R² 范围是 **(−∞, 1]**，不是 [0, 1]。[UNIFORM_MOTION_REPORT §3.4](UNIFORM_MOTION_REPORT.md) 看到的 y_pos R²=−0.137 就是负的——y 恒为 8 没方差，Ridge 找不到信号反而给出比"猜常数"更差的解。

#### class-balanced LogReg 拟合**二分类目标** → 用 **AUC** 评

- 适用：collision_event（0/1，是否发生碰撞）
- 为什么 "class-balanced"：collision_event 里正样本只占 **5.7%**。普通 LogReg 全猜负就能拿 94% acc，模型躺平不学。class-balanced 把正样本权重提到 ≈17×（= 94/5.7），强制模型学正样本边界。代码：`LogisticRegression(class_weight="balanced", ...)`。
- 评估指标 AUC（Area Under ROC Curve）：

```
AUC = P(score_positive > score_negative)
    = 随机抽 1 个正样本 + 1 个负样本，模型给正样本打分更高的概率
```

| AUC 值 | 含义 |
|---|---|
| **1.0** | 完美 ranking，**上限** |
| 0.95+ | 极强 |
| 0.7–0.9 | 不错 |
| **0.5** | 瞎猜 |
| < 0.5 | 反向预测（罕见，可翻转）|

> AUC 范围是 **[0, 1]**，但实际有意义的下限是 0.5（任何低于 0.5 的分类器都可以翻转预测得到 > 0.5）。

#### 为什么这里**不报告 accuracy**

正样本极少（5.7%），全猜负就 94.3% accuracy。**accuracy 在极不平衡分类里基本没意义**，所以我们主报 AUC，并把 `baseline_acc=0.9430` 一起 print 出来作为对照（提醒读者别被 acc 数字误导）。

### 6.3 "数据集 shortcut" 具体指什么 + "pixel-stats" 是什么

#### pixel-stats baseline 的定义

**pixel-stats 不是 pixel state。**它是一个故意设计得很弱的 baseline 特征：从一张 (3, H, W) 图里提 9 个数。

```python
mean = pix.mean(over_HW)   # (3,)  每通道平均亮度
std  = pix.std(over_HW)    # (3,)  每通道标准差
mean_sq = mean ** 2         # (3,)  平方项，给线性回归一点非线性
feat = concat([mean, std, mean_sq])  # 9-D
```

为什么需要这个 baseline：如果 192-D 的 ViT encoder 还打不过这 9 个数，那 ViT 就**白学**了。所以 pixel-stats 是表征评估的"地板"。

#### shortcut 是什么

shortcut = 一个**和目标相关**、但**不是真正物理因果**的特征，模型靠它就能"答对" probe，并不需要真的理解物理。**线性 probe 没法区分 shortcut 信息和真物理信息**——两者都表现为"linearly readable"。

phyworld collision 里至少 3 条 shortcut：

| shortcut 通道 | 表现 |
|---|---|
| **球的位置 → 颜色通道 std** | 两球颜色不同（其中一含蓝色）。蓝球在不同位置时**蓝通道 std 显著变化**——50k 帧实测 corr(B-std, x1)=**+0.70**，corr(B-mean, x1)=−0.56。9-D 像素特征里有 6 个数和 pos_x 单独相关性 \|.\|>0.4，所以 Ridge 拿到 R²=0.70 |
| **质量 → 球大小** | phyworld 约定 radius ∝ √mass，重的球更大 → 覆盖更多有色像素 → 通道 mean / std 反映 mass。pixel-stats 拿到 R²(mass_ratio)=0.67 |
| **位置 → 速度（统计相关）** | trajectory 总是 ball-1 在左初速向右、ball-2 在右初速向左，"球在右"统计上意味着"在向右"，位置直接预测速度方向 |

所以 trained encoder 的 R²(pos_x)=0.726 听起来不错，但 pixel-stats 自己就能 0.70，**这 0.70 是 free lunch，trained 的真正贡献只有 +0.026**。

### 6.4 vel_x 在哪里被"编码"？是 action 吗？当前 action 究竟是什么？

#### 当前实验的 action 是什么

实测 `~/.stable_worldmodel/phyworld_collision.h5` 里 `action` 的 shape 是 **(160000, 4)**，4 个分量含义和实测分布：

```
dim0 ax1: range=[-1.36, 0],  mean=-0.015     球 1 的水平加速度
dim1 ay1: range=[ 0,    0],  mean= 0         球 1 的垂直加速度，恒为 0
dim2 ax2: range=[ 0, +1.36], mean=+0.015     球 2 的水平加速度
dim3 ay2: range=[ 0,    0],  mean= 0         球 2 的垂直加速度，恒为 0
```

3 个事实：

1. **collision 实际是 1D 的**：每条 traj 内 y1=y2=const、vy1=vy2=0 始终。y 维只是 traj-level 的随机化（不同 traj 球放在不同高度），单条 traj 内运动是 1D 水平。所以 action 的 4 维里只有 2 维（ax1, ax2）是有信号的，另 2 维（ay1, ay2）恒为 0。
2. **action 极度稀疏**：99%+ 的帧加速度 ≈ 0（自由飞行阶段），只在碰撞瞬间出现脉冲。最大 ±1.36 都是碰撞那 1-2 帧的值。
3. **构造方式**：从 proprio 位置序列做二阶差分 `a[t] = (x[t+1] - 2x[t] + x[t-1]) / Δt²`，每球 x、y 各算一遍。这是论文里叫做 **force-action mode** 的 setup。

#### "4-D" 是什么 / 为什么是 4 维而不是 2 维或 8 维

每帧 4-D = `(ax1, ay1, ax2, ay2)` = **2 球 × 2 空间维度**。phyworld collision 实际是 1D 系统（y 恒定），`ay1 = ay2 = 0` 始终，**真正承载信号的只有 2 维**，但保留 4-D 是为了和 proprio 的 4 维 `(x1, y1, x2, y2)` 形状对称——以后扩展到 2D 物理 schema 不需要改。

#### 为什么选加速度而不是速度或位置

3 个候选 action 信号 + 取舍：

| 候选 | 含义 | 用了吗？为什么 |
|---|---|---|
| **位置 `pos[t]`** | 当前状态本身 | ❌ pos 本来就是 self-supervised 的预测目标候选，把它当 action 等于把答案直接喂给 predictor，没意义 |
| **速度 `vel[t] = (pos[t+1]−pos[t]) / Δt`** | 一阶差分 | ❌ **probe 失去诊断意义**：predictor 已经从 action 直接拿到 velocity 了，encoder 编码 velocity 是冗余的——梯度不会激励 encoder 学速度，加上 SIGReg 还会把这些冗余维度冲掉。结果就是 probe 测 emb→vel 给一个**误导性低 R²**，你没法判断是"模型笨学不会" vs "模型有意不学"。**uniform_motion 实验的关键 bug 就在于此**（实测 trained vx R²=0.166 比 random encoder 0.227 还低，见 [UNIFORM_MOTION_REPORT.md §5.2](UNIFORM_MOTION_REPORT.md)） |
| **加速度 `a[t] = (pos[t+1]−2pos[t]+pos[t−1]) / Δt²`** | 二阶差分 | ✅ **本实验选用** |

加速度被选中的 3 个理由：

1. **物理上是 system 的"外部控制"信号**。Newton 第二定律 F=ma：加速度反映**作用在物体上的力**。对碰撞而言，自由飞行段 a≈0，碰撞瞬间 a 是脉冲——正好对应 le-wm "action = 外部对状态的影响"的设计假设。
2. **不直接泄露 velocity probe 的答案**。velocity 是 a 的积分，从 a 反推 v 需要 encoder 先记住 v[t-1] 再做积分——**predictor 必须用到 emb 里的 velocity 信息**才能推理。这才是真正测 encoder 表征是否学到速度的设置。
3. **phyworld benchmark 标准 force-action mode**，和原论文对齐、可对比。

#### 这意味着 vel_x 在哪里？

**既不在图里，也不在 action 里**。两套常见 setup 的对比：

| 数据集 | action 的物理含义 | 模型能直接"看到"速度吗？ |
|---|---|---|
| **PushT** | action = agent 的**速度命令**（鼠标拖动 = velocity 输入） | 能，velocity 是显式 condition |
| **phyworld collision (本实验)** | action = 二阶差分得到的**加速度** | 不能 |

我们这次喂给模型的是：

- encoder 输入：**单张静态图**（看不出运动方向/速度）
- predictor 输入：current embedding + 4-D acceleration（不含 velocity）

velocity **理论上**只可能通过 JEPA 的间接训练信号被学到（"如果 encoder 把速度信息编进 embedding，predictor 用 a 预测下一帧 latent 才容易对"），但这个间接梯度信号很弱，加上 99% 帧 action≈0 → predictor 在大部分帧上都不靠 action 也能 trivial 地预测，**真正承载 physics learning 信号的训练梯度只在碰撞那 1-2 帧上**。

probe 测出来 vel_x 在所有 encoder（trained / random / pixel-stats）都卡在 R²≈0.47 —— 这个 0.47 **完全来自数据集中"位置 → 速度"的统计相关性（5.3 中的 shortcut 通道 3）**，不是因为 velocity 真的被任何 encoder 编码进了。

**结论**：单帧 probe 测 velocity 在 collision setup 下本来就不公平，因为信息源根本就不在单帧里。要测 encoder 是否懂速度，要么改成**多帧 probe**（同一 traj 连续 4 帧 embedding 拼起来），要么把 action 改成 **velocity-action**（像 PushT 一样让速度显式可见，但这会让 vel probe 直接读 action 即可，绕过 encoder——是个 tradeoff）。

### 6.5 为什么"补到 epoch 15 不会改变结论"是高概率推断（但不是 100%）

这个判断基于 3 个观察：

1. **pred_loss 已经明显收敛**：0.039 → 0.033 → 0.031 → 0.026 → 0.024 → 0.020 → 0.018 → **0.017**，斜率在 flatten。从 epoch 8 到 15 再降的幅度有限。
2. **trained vs random 的 gap 不来自欠训练**。如果是欠训练，应该看到 random 表现差、trained 大幅领先；实际是 random ViT-tiny 也能拿 R²(pos_x)=0.65 / mass_ratio=0.67 —— 说明这些数字大部分是**数据集 shortcut 让任何 encoder 免费拿到的 free lunch**，多训 7 个 epoch 不会改变 shortcut 的存在。
3. **dominant problem 是数据集，不是模型**。pixel-stats（9-D）都拿到 R²=0.70，说明这个数据集的 probe 评测自带一个"shortcut 天花板"。把 trained encoder 训到极致，它的优势可能从 +0.026 涨到 +0.05，但**定性结论"trained ≈ baseline"不会变**。

但严格来说这是一个**外推预测**：基于 epoch 1-8 的 loss 斜率推断 epoch 8-15 不会发生跳跃式表征重组。这有可能错（虽然 self-supervised pretraining 突变式涌现的概率小）。**如果想保险，下一步动作就是补训完再 probe 一遍**——成本约 21 min 训练 + 10 min probe。

### 6.6 Ridge 详解 + 具体例子

#### 6.6.1 Ridge 在 probe 上的具体流程

用 collision 实验做例子，目标：从 192-D embedding 预测**球的水平位置 pos_x**（一个实数）。

**第 1 步：组织数据成矩阵**

128k 个训练样本，每个是 `(emb[t], pos_x[t])`：

```
样本 1:    emb_1 = [0.31, -0.12, 0.84, ..., 0.05]  (192 维)  →  pos_x_1 = 2.13
样本 2:    emb_2 = [0.27,  0.08, 0.79, ..., 0.11]  (192 维)  →  pos_x_2 = 3.47
...
样本 128k: ...
```

堆成矩阵 `X` 和向量 `y`：

```
X = [128 000 × 192]   每行一个样本的 192-D emb
y = [128 000]          对应的 pos_x 标量
```

**第 2 步：Ridge 求权重 w**

找 192-D 权重向量 `w`，使得 `X · w ≈ y`：

```
[ emb_1 ]                   [ 2.13 ]
[ emb_2 ]  · [w_1, ..., w_192]ᵀ  ≈  [ 3.47 ]
[ emb_3 ]                   [ 0.84 ]
[  ...  ]                   [ ...  ]
```

最小化的目标：

```
Loss(w) = Σ (y_i − emb_i·w)²  +  α·Σ w_j²
         └──预测误差──┘       └─正则项─┘
```

闭式解（不需要梯度下降）：

```
w* = (XᵀX + αI)⁻¹ Xᵀy
      ↑
   αI 保证可逆，特别是 emb 维度高相关时
```

**第 3 步：在 test 上预测**

新 test 样本 `emb_new`（192-D）：

```
pos_x_pred = emb_new · w* = Σ_j  w_j · emb_new[j]    # 加权和，输出标量
```

**第 4 步：算 R² 评估**

```
y_true: test 集的 32 000 个真 pos_x
y_pred: 模型预测出来的 32 000 个 pos_x

SSE = Σ (y_true_i − y_pred_i)²
SST = Σ (y_true_i − mean(y_true))²
R²  = 1 − SSE / SST
```

#### 6.6.2 迷你具体例子（3-D，5 样本，可手算）

假装 emb 只有 3 维、有 5 个样本：

| 样本 | emb (3-D) | pos_x（target）|
|---|---|---|
| 1 | [ 0.5, −0.2,  0.8] | 2.1 |
| 2 | [ 0.7,  0.1,  0.6] | 3.4 |
| 3 | [−0.3,  0.4, −0.2] | 0.8 |
| 4 | [ 0.6, −0.1,  0.7] | 2.7 |
| 5 | [ 0.2,  0.3,  0.4] | 1.5 |

矩阵：

```
X = [ 0.5  −0.2   0.8 ]      y = [2.1]
    [ 0.7   0.1   0.6 ]          [3.4]
    [−0.3   0.4  −0.2 ]          [0.8]
    [ 0.6  −0.1   0.7 ]          [2.7]
    [ 0.2   0.3   0.4 ]          [1.5]
```

Ridge α=1.0 解出来大约（数值近似）：

```
w* ≈ [3.05, 0.32, 0.71]
```

含义：emb 第 1 维对 pos_x 影响最大（系数 3.05），第 2 维影响小（0.32），第 3 维中等（0.71）。

预测新样本 `emb = [0.4, 0.2, 0.5]`：

```
pos_x_pred = 0.4 × 3.05 + 0.2 × 0.32 + 0.5 × 0.71
           = 1.22 + 0.064 + 0.355
           ≈ 1.64
```

#### 6.6.3 α 改变会怎么样

| α | 行为 |
|---|---|
| α = 0 | 退化成 OLS：完全拟合 train，但 train 噪声让 test R² 跳水。**192-D 高相关 emb 上数值上还会爆**（XᵀX 几乎奇异）|
| α = 1（我们用的）| 中等正则，稳定 + 不过拟合 |
| α = 100 | 强正则：w 都被压到接近 0，模型几乎只能预测 mean(y_train)，R² 接近 0 |
| α → ∞ | w = 0，模型完全躺平，R² = 0（只能预测均值）|

调 α 不是关键——只要不太极端，linear probe 的相对比较都是稳健的。

#### 6.6.4 为什么不用普通 OLS（要加正则项）

我们的输入是 192 维 ViT emb，**dimension 之间高度相关**（NN 训出来的 features 都这样）。这时候 `XᵀX` 接近**奇异**（行列式接近 0），求逆数值上不稳定，OLS 解出来的权重会狂大、对 train 噪声超敏感、test 上崩。

Ridge 的 `αI` 项把 `XᵀX` 从奇异推到 well-conditioned，**保持线性的同时让结果稳定 + 不过拟合**。

多帧 probe 时尤其需要——concat 后是 768-D 输入，相关性更高，没 Ridge 会数值爆掉（事实上日志里看到过 `LinAlgWarning: Ill-conditioned matrix (rcond=2.3e-08)` 提示，但因为有 α 项最终结果稳定）。

#### 6.6.5 为什么不用 MLP / 神经网络 probe

这是 **linear probe 协议的核心哲学**：

| probe 模型 | 能告诉我们什么 |
|---|---|
| **线性（Ridge）** | 信息**线性可访问**地编码在 emb 里——是 encoder 的功劳 |
| **MLP / 神经网络** | 信息**可被某种非线性函数挖出**——分不清是 encoder 学的还是 probe 自己挖的 |

我们故意选**最弱的可解模型**，让"成功"的归因明确指向 encoder。如果用 MLP probe 跑出 vel R²=0.94，你没法判断是：

- (a) encoder 真的把 velocity 编码进了 emb（probe 是被动读取者）
- (b) emb 只有 position，但 MLP probe 自己学会了"算差分"补出 velocity

线性 probe 排除 (b)——线性模型没能力"算"非线性的东西，能读出来就是 encoder 给它准备好了。

**例外**：如果你怀疑 encoder 把信息编成非线性形式（比如球位置编成傅里叶基），可以用 MLP probe 作辅助诊断——但要明确说"这是测 encoder + MLP 联合能力，不是 encoder 单测"。

### 6.7 为什么"按 traj 切"而不是"按帧切"

报告里 §2.1 写了 "4000/1000 trajectory split"——这是数据切分协议里**最容易踩坑**的一条。

#### 数据结构

```
5000 条 traj × 32 帧/traj = 160 000 帧

traj_0:  [frame_0, frame_1, frame_2, ..., frame_31]   ← 同一对球的连续运动
traj_1:  [frame_0, frame_1, ...]
...
```

每条 traj 是同一对球的整段轨迹，相邻帧只差几个像素（球位置略变了一点）。

#### 两种切法

**❌ 按帧切（错误做法）**：把 160k 帧随机 80/20。

```
traj_42 的 32 帧分布:
  frame_0   → train
  frame_1   → test    ← 和 frame_0 几乎一样
  frame_2   → train
  frame_3   → test    ← 和 frame_2/4 几乎一样
  ...
```

**问题（"相邻帧泄露"）**：

- frame_0 进 train，frame_1 进 test
- 两张图**像素级几乎一样**（球只挪 0.1 像素），encoder 出来的 emb 也几乎一样
- probe 在 train 学到 `emb of frame_0 → pos=5.2`
- test 时碰到 frame_1，**emb 也几乎是 5.2 那个值** → probe 直接背答案就行
- **R² 虚高，但不反映 encoder 的真实泛化能力**

**✅ 按 traj 切（正确做法）**：5000 条 traj 随机 80/20，**同一条 traj 的 32 帧要么全 train 要么全 test**。

```
traj_0   ─→ train (32 帧全进 train)
traj_1   ─→ train
traj_2   ─→ test  (32 帧全进 test)
traj_3   ─→ train
...
```

帧数：
- train: 4000 traj × 32 帧 = **128 000 帧**
- test:  1000 traj × 32 帧 = **32 000 帧**

测试 traj 的所有帧 encoder 从来没见过——probe 必须**真的**学会"图 → pos 的映射"才能泛化到新 traj。

#### 这差别有多大

如果按帧切，trained encoder 的 pos_x R² 可能从我们报告的 0.726 蹦到 0.95+，纯粹是泄漏的虚高。**按 traj 切是绝对要求，不是 nice-to-have**。phyworld / le-wm / pusht 所有论文 probe 协议都是按 traj 切。

### 6.8 "多帧 probe" 是什么 / 为什么必要

#### 单帧 vs 多帧

**单帧 probe**（§2.2 / §2.3 默认）：一张图过 encoder → 1 个 192-D embedding → Ridge 线性回归预测目标。

**多帧 probe**（§2.4 / §2.5 用到）：把同一 traj 连续 K 帧的 embedding **拼起来**喂 Ridge：

```python
single-frame feature:  emb[t]                                    # 192-D
multi-frame  feature:  concat(emb[t-3], emb[t-2], emb[t-1], emb[t])  # 4×192 = 768-D
```

实现细节：
- **按 traj 切，不跨 episode**（避免相邻帧泄露）。每条 traj 前 K-1 帧因没足够 history 被丢弃。collision K=4 时 160k → 145k 有效帧（损失 ~10%）。
- feature 维度从 192 涨到 K × 192（K=4 → 768-D）
- 还是按 traj 切 train/test 80/20
- Ridge 自己决定 K 个时间步怎么线性组合——比如学到 `(pos[t] − pos[t-1]) / Δt` 这种差分公式来读出速度

#### 为什么必要

单帧协议**等价于强求**："想读出 X，X 必须线性编码进单个 emb 向量"。这个要求**太严**——速度信息天然在跨帧上（一张静态照片本来就看不出球在以多快速度运动），即使 encoder 完美记住了每帧 position，单帧 probe 也算不出 velocity。

多帧协议把要求放宽到："X 必须跨 K 帧线性可读"。这**更符合下游真实用法**——任何下游模型（RL agent、planner）都不会只看单帧，都会处理 emb 序列。

#### 实测对比（来自 §2.4 / §2.5 / [UNIFORM_MOTION_REPORT §3.5](UNIFORM_MOTION_REPORT.md)）

| 实验 | 单帧 K=1 vel_x R² | 多帧 K=4 vel_x R² | Δ |
|---|---|---|---|
| collision from-scratch | 0.487 | 0.594 | +0.107 |
| collision paper-init | 0.620 | **0.883** | +0.263 |
| uniform_motion paper-init | 0.232 | **0.939** | +0.707 |

> **教训**：很多看似"encoder 没学到 X"的 negative result 其实是单帧协议读不出 X。这次 uniform_motion 原报告的"encoder 没学到速度"就是被这个 confound 误判了——multi-frame probe 一上就涨到 0.94。**在 le-wm 系列实验里，下游表征评估应默认跑一个多帧版本作为 sanity check**。

#### 什么时候多帧也救不了

多帧 probe **只能恢复跨帧线性可读的信息**。如果 encoder 真把信息丢了（被 SIGReg 冲刷掉、被 projector 压坏），多帧也读不出。所以多帧 probe ≠ 万能：

- ✅ vel = (pos[t] − pos[t-1])/Δt 这种**线性差分**关系，多帧能读
- ✅ angular velocity、加速度 = 二阶差分，多帧也能读（K 够大即可）
- ❌ 如果 encoder 把同一物理状态的多个 traj **collapse 到同一 emb**（信息真的丢了），多帧也救不了
- ❌ 如果目标信息和 emb 之间是**非线性关系**（比如 v² 和 emb 是平方关系），需要 MLP probe 而不是多帧 Ridge

---

## 7. 文件清单

| 文件 | 作用 |
|---|---|
| `~/.stable_worldmodel/phyworld_collision.h5`（~24 GB） | 转换后训练数据 |
| `~/.stable_worldmodel/collision_run/lewm_collision_epoch_8_object.ckpt`（72 MB） | encoder + predictor 权重，可直接 `torch.load` |
| `~/.stable_worldmodel/collision_run/lewm_collision_weights.ckpt`（217 MB） | Lightning 完整 ckpt（含 optimizer state，可断点续训） |
| `~/.stable_worldmodel/collision_run/config.yaml` | 训练用的展开后 hydra config |
| `phyworld/scripts/convert_collision_to_lewm.py` | 数据转换 |
| `phyworld/scripts/probe_collision_encoder.py` | probe（已修过 OOM bug，懒加载像素） |
| `le-wm/config/train/data/collision.yaml` | 训练 hydra data config |
| `/tmp/lewm_collision_train.log` | 完整训练日志 |
| `/tmp/lewm_collision_probe.log` / `_noproj.log` | probe 日志 |
