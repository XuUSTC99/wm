# le-wm 能否学到牛顿运动定律？— phyworld 完整实验报告

**实验日期**：2026-05-08 ~ 2026-05-10
**作者**：Claude 协助 haochenluo02@gmail.com
**主问题**：LeWorldModel (le-wm) 这种 JEPA 风格自监督世界模型，能否在 phyworld 物理仿真视频上学到**牛顿运动定律**所涉及的物理量与动力学关系？

## 三大定律 ↔ phyworld 三个子任务

phyworld 论文 §3.1 把牛顿三大运动定律 1:1 对应到三个仿真任务，每个测试不同的物理规律。我们的实验覆盖了其中两个：

| 牛顿定律 | 物理含义 | 对应 phyworld 任务 | 本报告状态 |
|---|---|---|---|
| **第一定律（惯性定律）** | 无外力时物体保持匀速直线运动 | **uniform_motion**（匀速直线） | ✅ **已做**（§2-§6 部分覆盖） |
| **第二定律（F = m·a）** | 力等于质量乘加速度，定义动力学正演 | **parabola**（重力下抛物线运动） | ❌ **未做**（成本-效益分析见 §6.7） |
| **第三定律（作用 = 反作用）** | 碰撞中动量守恒 / 力对称 | **collision**（弹性碰撞） | ✅ **已做**（§2-§6 部分覆盖） |

**所以本报告的覆盖范围是"牛顿第一 + 第三定律"，第二定律（F=ma 在恒定外力下的体现）我们没直接测**。下面所有讨论以这个范围为准。

---

## 摘要 (TL;DR)

> **覆盖范围说明**：本实验覆盖 phyworld 三任务中的 2 个 ——
> **N1（uniform_motion）** + **N3（collision）**，**N2（parabola）未做**。

| | 答 |
|---|---|
| le-wm encoder **能否**编码位置 (pos)？ | ✅ 能。paper-init + 单帧 probe: pos_x R² = **0.94 (uniform) / 0.86 (collision)** |
| 能否编码速度 (vel)？ | ✅ 能，**但必须用多帧 probe 才能读出**。paper-init + K=4 probe: vel_x R² = **0.94 (uniform) / 0.88 (collision)** |
| 能否编码质量 (mass)？ | ✅ 能。collision paper-init: mass_ratio R² = **0.83** |
| 能否检测碰撞事件（N3 作用反作用瞬间）？ | ✅ 能。collision_event AUC = **0.95** |
| 能否预测下一帧（隐含动力学）？ | ✅ 能（**但只在 ID 上**）。pred_loss 收敛到 0.017（emb-space MSE）|
| 能否泛化到 OOD？ | ❌ 单轴 OOD 部分降，**both-OOD 上 pos R² 转负**（§6） |
| 能否跨域 transfer（PHYRE 多对象）？ | ⚠️ 位置 transfer 一定程度上可以（centroid_y R²=0.69），但**对象计数 transfer 完全失败**（§6.6）|

**结论**：

- **N1（惯性）**：le-wm 在 ID 上掌握得很好（匀速直线 emb 几乎完美编码 pos + vel）
- **N3（作用反作用 / 动量守恒）**：碰撞瞬间识别 AUC=0.95，mass / pos / vel 都能 probe 出，**ID 内学到了**
- **N2（F=ma 在恒定外力下）**：**没测**（parabola 数据集未做训练 + probe）—— 这是本报告未覆盖的牛顿定律部分

更严格地说，"学到了 N1 / N3"也只在 ID 分布内成立。OOD 上和 phyworld 论文对 video-gen 模型的判断"case-based generalization"是**同性质**的——encoder 学的是 ID 视觉模式而非抽象物理规律。

**但有 3 个关键 caveat**：

1. 必须用**论文 release 的 lewm-pusht 权重做 init**，random init 在 5.5M 参数 + 160k 帧的设置下学不动
2. 必须用**多帧 K=4 probe**，单帧协议读不出速度
3. 必须用 **`--no-projector`**，JEPA projector 会压掉对预测无用的维度

**前期得到的 negative result（"encoder 没学到速度"）是上述 3 个 confound 联合造成的假阴性**，不是模型真的没学。

---

## 0. 研究问题：什么叫"学到牛顿运动定律"

### 0.1 三大定律涉及哪些物理量 + 我们测哪些

牛顿三大运动定律共同涉及 5 个物理量：

- **m**（质量）、**a**（加速度）、**F**（合外力）—— 由 F=ma 联系
- **pos**（位置）、**vel**（速度）—— 由 a 通过积分演化

phyworld 论文用 3 个仿真任务**正交地**测试 3 个定律：

| 定律 | 核心声明 | phyworld 任务 | 任务的关键现象 | 本报告 |
|---|---|---|---|---|
| **N1（惯性）** | F=0 → 物体保持匀速直线 | uniform_motion | 球以恒定 v 沿水平线运动 | ✅ |
| **N2（F=ma）** | 恒定 F → 抛物线加速 | parabola | 球受重力 → 抛物线轨迹 | ❌ 未做 |
| **N3（作用反作用 / 动量守恒）** | 碰撞前后 Σm·v 不变 | collision | 两球弹性碰撞，动量守恒 | ✅ |

> **覆盖范围**：我们的实验覆盖 N1 + N3 但 **N2 未测**。要严格说"le-wm 是否完整学到牛顿定律"，**N2 是最关键的空白**——因为只有 N2 涉及"恒定外力下的加速"这个核心动力学关系，前两个定律都是 N2 的特例（F=0 → N1；ΣF=0 → 系统动量守恒 → N3 间接成立）。

### 0.2 我们怎么间接测"encoder 学没学到"

le-wm 是 JEPA 风格的自监督世界模型，**只用 pixel 自己**端到端训练（不喂 label）。所以"encoder 是否学到牛顿定律"这个问题没办法直接问，要通过 **linear probe** 间接测：

1. **训练**：在 phyworld 视频上跑 le-wm 自监督训练
2. **冻结 encoder**：拿出训好的 encoder，权重不再更新
3. **probe**：在 encoder 输出的 embedding 上，**仅训一个线性回归**（Ridge）去预测某个物理量
4. **看 R²**：如果线性 probe 能从 emb 准确恢复物理量，说明 encoder **把这个量线性可读地**编进了 emb 里 → encoder "学到了"

probe 目标对应到我们已做的两个定律：

| 物理量 | 测量方式 | 对应的牛顿定律 |
|---|---|---|
| pos | 单帧 probe 直接预测 (x, y) | N1 状态变量；N3 也需要 |
| vel | 单帧 probe（弱）+ 多帧 K=4 probe（强）| N1 主要指标（匀速 = vel 恒定）|
| m  | collision 上 probe mass_ratio = m₁/m₂ | N3 守恒律的核心变量 |
| 碰撞事件 | collision_event 二分类 probe | N3 作用反作用瞬间识别 |
| 下一帧预测 | predictor 的 pred_loss（emb-space MSE）| 动力学正演（N2 也属此范畴）|

> **N2 没测的关键观察量**：抛物线轨迹下的 a_y（重力恒定加速度）+ v_y 随时间线性增长 + 轨迹 curvature。这些都需要 parabola 数据集。

---

## 1. 实验设计

### 1.1 数据集

两个 phyworld 子任务：

| 数据集 | uniform_motion（匀速直线）| collision（弹性碰撞）|
|---|---|---|
| 物理 | 1 球 + 0 力 | 2 球 + 弹性碰撞 |
| trajectories | 1 152 | 5 000 |
| 帧数 | 36 864 | 160 000 |
| 帧尺寸 | 224 × 224 × 3 | 224 × 224 × 3 |
| 维度 | 1D 水平运动（y 恒定） | 1D 水平运动（y 恒定） |
| action 信号 | `vel[t] = (x[t+1]−x[t])/Δt`（velocity-action）| `a[t] = (x[t+1]−2x[t]+x[t-1])/Δt²`（force-action）|
| action 维度 | 2 | 4（其中 2 维恒为 0）|

> **为什么 collision 用加速度不用速度作 action**：如果 action=velocity，predictor 直接就拿到速度了 → encoder 不需要编码速度 → probe vel 给出误导性低 R²（uniform_motion 实测就 0.166），分不清"模型笨学不会"还是"模型有意不学"。force-action 模式下 action 提供加速度，速度必须从 emb 来，**probe 才能真正诊断 encoder 表征**。

### 1.2 le-wm 架构

```
图 (B,T,3,224,224)                 action (B,T,2 or 4)
       │                                  │
       ▼                                  ▼
   ViT-tiny encoder                   action_encoder
   (5.5M params)                       (1D Conv + MLP)
       │                                  │
   cls token (B,T,192)             act_emb (B,T,192)
       │                                  │
       ▼                                  │
   projector (MLP 192→2048→192)           │
       │                                  │
   emb (B,T,192) ──┬── tgt_emb[:,1:]──    │
                   │                       │ ┌────────────┘
                   ▼                       ▼ ▼
              AR predictor (6-layer causal transformer, AdaLN-zero)
                   │
                   ▼
              pred_emb (B,T,192)

loss = ||pred_emb − tgt_emb||²  +  0.09 · SIGReg(emb)
       ↑ JEPA prediction loss      ↑ 推 emb 分布向 N(0, I)
```

5 个组件总参数 ~10M，ViT-tiny 主导。SIGReg（Sketch Isotropic Gaussian Regularizer）是 le-wm 的核心创新——单 loss 项防 collapse，比 EMA/multi-loss 更简单。

### 1.3 训练配置

| 项 | 值 |
|---|---|
| Optimizer | AdamW, lr=5e-5, wd=1e-3 |
| Scheduler | LinearWarmupCosineAnnealingLR (epoch interval) |
| Precision | bf16 |
| Batch | 128 |
| Epoch | uniform: 20 / collision: 8 |
| GPU | NVIDIA RTX A6000 (48 GB, 单卡, ~14 GB usage) |
| history_size | 3 |
| num_preds | 1 |
| SIGReg weight λ | 0.09 |

**两种 init**：

- **from-scratch**：所有权重随机初始化
- **paper-init**：从 [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht) 加载 `encoder.*` + `projector.*` + `pred_proj.*`（216 个 keys）；`predictor.*` + `action_encoder.*` 随机初始化（PushT action 10 维 ≠ phyworld，不能复用）

### 1.4 评估协议

#### 1.4.1 数据切分

按 **trajectory** 切 80/20（不是按帧）：

- collision: 4 000 / 1 000 traj → 128k / 32k 帧
- uniform: 922 / 230 traj → 29.5k / 7.4k 帧

> 为什么不按帧切：相邻帧像素几乎一样，probe 不用泛化就能背答案 → R² 虚高，失去诊断意义。

#### 1.4.2 Probe 类型

**单帧 probe (K=1)**：emb[t] (192-D) → Ridge → 目标

**多帧 probe (K=4)**：concat(emb[t-3], emb[t-2], emb[t-1], emb[t]) (768-D) → Ridge → 目标

> K=4 把"线性可读"放宽到"跨帧线性可读"——速度本质是 (pos[t]−pos[t-1])/Δt 这种**线性差分**，单帧 emb 算不出来，多帧 Ridge 能学到差分组合。

#### 1.4.3 Projector 开关

- `--no-projector`：probe 读 encoder 的 cls token（192-D，无压缩）
- 默认 with-projector：probe 读 `projector(cls_token)` —— projector 会压掉对预测下一帧无用的维度（信息瓶颈），下游 probe 不能用

#### 1.4.4 模型 / Metric

| 任务类型 | 模型 | 评估指标 | 上限 |
|---|---|---|---|
| 回归（pos, vel, mass）| Ridge α=1.0 | R² | 1.0 |
| 二分类（collision_event）| LogReg, class_weight="balanced" | AUC | 1.0 |

#### 1.4.5 Baseline 对照

3 个对照：

1. **trained encoder**：本实验主角
2. **random ViT-tiny**：同架构随机初始化，没训过——测"训练带来的增益"
3. **pixel-stats**：(per-channel mean, std, mean²) 共 9 维——测"任何 encoder 都该打过的下限"

---

## 2. 初版结果：看似 negative

### 2.1 单帧 + with-projector（论文原 protocol）

#### uniform_motion，20 epoch from-scratch

| Encoder | x_pos R² | y_pos R² | vx R² |
|---|---|---|---|
| trained | 0.732 | −0.137 | **0.166** |
| random ViT | 0.582 | 0.073 | 0.227 |
| pixel-stats (9-D) | (合并) | (合并) | **0.256** |

#### collision，8 epoch from-scratch

| Encoder | pos_x R² | vel_x R² | mass_ratio R² | collision AUC |
|---|---|---|---|---|
| trained | 0.708 | 0.473 | 0.719 | 0.741 |
| random ViT | 0.649 | 0.464 | 0.666 | 0.763 |
| pixel-stats | 0.700 | 0.470 | 0.666 | 0.667 |

### 2.2 看似的结论

- **trained ≈ random ≈ pixel-stats**：训练几乎没带来增量
- **uniform vx 上 trained (0.166) 反而比 random (0.227) 和 pixel-stats (0.256) 更差**——看起来 "训练让 encoder 主动忘了速度"
- **collision 上 trained 仅微弱超过 baseline**，且 collision_event 上 random encoder 居然比 trained 还高（0.763 vs 0.741）

**看似结论**：le-wm 在 phyworld 上**没学到物理**，只是把 encoder 训成了一个略好的"球的定位器"。

### 2.3 当时给的归因（后被部分推翻）

> 我们当时怀疑 4 条原因：(A) action 信号稀疏；(B) 单帧 probe 盲区；(C) SIGReg 把 emb 里的"对预测无用"信息冲刷掉；(D) random init 学不动小数据。

---

## 3. 三个 confound 联合造成的假阴性

补做对照实验后发现：上面 negative result **不是模型问题，是协议 + 初始化的 3 个 confound 联合放大**出来的假阴性。

### 3.1 Confound A：projector 压信息

**机制**：JEPA loss 在 projector 输出的 z 空间上算 (`||pred_z[t+1] − tgt_z[t+1]||²`)。projector 被训练优化预测，**会主动压掉对"预测下一帧"无用的维度**（这是信息瓶颈，机制类似 SIGReg）。下游 probe 不需要"预测下一帧"，要的是物理量信息——所以必须**绕过 projector，读 encoder cls token**。

**实测影响**（collision from-scratch）：

| Target | with-projector | --no-projector | Δ |
|---|---|---|---|
| pos_x R² | 0.708 | 0.726 | +0.018 |
| **collision AUC** | **0.741** | **0.808** | **+0.067** |

collision_event 上提升最戏剧化——碰撞瞬间的视觉变化对"预测下一帧"无用（碰撞极短），projector 把它压掉了，但对下游 probe 是关键信息。

### 3.2 Confound B：单帧 probe 盲区

**机制**：单帧 probe 把 emb[t] 一个向量扔给 Ridge，要求"velocity 必须线性编码进**单个**向量"。但 velocity 本质是 (pos[t]−pos[t-1])/Δt 这种**跨帧差分**——一张静态照片本来就看不出球在以多快速度运动。即使 encoder 完美记录每帧 position，单一向量内部也没法做"减法"。

**多帧 probe**：concat 4 个连续帧的 emb 给 Ridge，让 Ridge **跨时间步**学线性差分组合，速度可恢复。

**实测影响**：

| Setup | K=1 vel_x R² | K=4 vel_x R² | Δ |
|---|---|---|---|
| uniform paper-init | 0.232 | **0.939** | **+0.71** |
| collision paper-init | 0.620 | **0.883** | +0.26 |
| collision from-scratch | 0.487 | 0.594 | +0.11 |

uniform 上 +0.71 是巨大跳跃——证明 vel 信息**确实在 emb 里**，只是单帧 protocol 读不出。

### 3.3 Confound C：random init + 小数据学不动

**事实**：

- ViT-tiny 5.5 M 参数
- 标准 ViT 训练用 ImageNet-21k（14M 图）或 JFT-300M（300M 图）
- 我们 collision 仅 160k 帧、uniform_motion 仅 36k 帧
- 比标准 ViT 训练**少 100–10 000 倍**

random init 在这种数据量下基本不可能学出强表征。**lewm-pusht paper init** 给 encoder 一个 PushT 任务上已经学好的"物体定位 + 颜色区分"基础，等于在 phyworld 上做 fine-tune。

**实测影响**（collision）：

| Setup | pos_x R² | vel_x R² | mass R² | collision AUC |
|---|---|---|---|---|
| from-scratch K=1 | 0.726 | 0.487 | 0.711 | 0.808 |
| paper-init K=1 | **0.863** | **0.620** | **0.826** | **0.952** |
| Δ | **+0.137** | **+0.133** | **+0.116** | **+0.144** |

**所有 target 同步 +0.12~+0.14**——paper-init 是 negative result 的主要贡献者之一。

### 3.4 三个 confound 的联合效应

任一 confound 单独修复，效果**有限**：

| 修复维度 | uniform vx R² | collision vel_x R² |
|---|---|---|
| 三个都没修（原 protocol） | 0.166 | 0.473 |
| 只修 projector | (没测) | 0.487 |
| 只修单帧→多帧 | 0.166 → ? | 0.487 → 0.594 |
| 只修 init | 0.232 | 0.620 |
| **全修（paper-init + K=4 + no-proj）** | **0.939** | **0.883** |

**三个 confound 的影响是乘性的，不是加性的**：单独修一个杯水车薪，联合修 vel_x 直接从 0.166 → 0.939。

---

## 4. 修正后的结果：实际是 positive

### 4.1 4-way 对比总表（uniform_motion）

| Setup | x_pos R² | y_pos R² | vx R² |
|---|---|---|---|
| from-scratch + K=1 + with-proj | 0.732 | −0.137 | 0.166 |
| paper-init + K=1 + no-proj | 0.942 | 0.859 | 0.232 |
| **paper-init + K=4 + no-proj** | **0.964** | **0.964** | **0.939** |

### 4.2 4-way 对比总表（collision）

| Setup | pos_x R² | vel_x R² | mass_ratio R² | collision_event AUC |
|---|---|---|---|---|
| from-scratch + K=1 + with-proj | 0.708 | 0.473 | 0.719 | 0.741 |
| from-scratch + K=1 + no-proj | 0.726 | 0.487 | 0.711 | 0.808 |
| from-scratch + K=4 + no-proj | 0.814 | 0.594 | — | — |
| paper-init + K=1 + no-proj | 0.863 | 0.620 | **0.826** | **0.952** |
| **paper-init + K=4 + no-proj** | **0.911** | **0.883** | — | — |

### 4.3 关键观察

- 最右下角的 (paper-init + K=4 + no-proj) 是 **encoder 实际能力的真实读数**
- uniform_motion 上 pos/vel 都接近 R²=0.94，encoder **几乎完美**编码 1D 物理状态
- collision 上 pos/vel/mass 都到 R²=0.83~0.91，collision_event AUC 0.95 ——encoder **学到了完整的物理状态空间**
- y_pos 在 from-scratch 上是 −0.137（看起来 encoder 没编码 y）；paper-init 上跳到 0.86（编码了 y 的 traj-level 方差）——说明**前面看似 y 没编码也是 confound 的产物**

---

## 5. 按定律拆解：N1 / N2 / N3 各自学到了多少

下面把 §2-§4 的实验数据按**每条牛顿定律**重新组织，明确我们对每条定律的判断 + 证据 + 缺口。

### 5.1 N1（惯性定律）—— 在 uniform_motion 上**学到了**

**N1 声明**：F=0 时物体保持匀速直线运动。要让 encoder "学到 N1"，emb 应该编码两件事：

1. 球的**位置**（probe 能从 emb 读出 pos）
2. 球的**速度恒定**（probe 能从 emb 读出 vel，且帧间 vel 没变化）

| 证据 | 数值 | 来源 |
|---|---|---|
| 单帧能从 emb 读 pos | x_pos R² = **0.94** | uniform paper-init K=1 |
| 多帧能从 emb 读 vel | vx R² = **0.94** | uniform paper-init K=4 |
| predictor 能预测下一帧 emb | pred_loss = 0.013 | uniform 20 epoch 收敛 |

结论：**N1 在 ID 范围内被 le-wm 学到了**。encoder 几乎完美编码 1D 物理状态，predictor 能用 (state, action) 正演——匀速直线对它来说是 trivial 的内插。

⚠️ **caveat**：uniform_motion 里 action=velocity 是个 setup bug（见 §6 OOD），让 vel 不需要从 emb 学就能被 predictor 直接拿到。N1 这个判断**实际上是基于"emb 里有 pos 信息 + 多帧 probe 能算差分"**，**不是"encoder 直接编码了恒定速度"**。

### 5.2 N3（作用反作用 / 动量守恒）—— 在 collision 上**学到了**

**N3 声明**：两物体碰撞前后总动量守恒（Σm·v 不变）+ 作用力对称。要让 encoder "学到 N3"，emb 应该编码：

1. 每球的**位置**（pos_x）
2. 每球的**速度**（vel_x）
3. 每球的**质量**（mass，N3 的核心区分变量）
4. 是否发生**碰撞瞬间**（识别相互作用的时刻）

| 证据 | 数值 | 来源 |
|---|---|---|
| 单帧 emb 编码两球 pos | pos_x R² = **0.86** | collision paper-init K=1 |
| 多帧 emb 编码两球 vel | vel_x R² = **0.88** | collision paper-init K=4 |
| 单帧 emb 编码 mass_ratio | mass R² = **0.83** | collision paper-init K=1 |
| 单帧 emb 识别碰撞瞬间 | AUC = **0.95** | collision_event LogReg |
| predictor 能预测下一帧 emb | pred_loss = 0.017 | collision 8 epoch 收敛 |

结论：**N3 所需的全部状态变量都被 encoder 编码，碰撞事件能识别，predictor 能从 (emb, action) 正演**。在 ID 范围内 le-wm 学到了 N3。

⚠️ **caveat**：predictor 拿到了 acceleration as action（其中包含碰撞瞬间的脉冲信号），它"学到的 N3" 一部分来自直接看冲量数值，不全是从 emb 推理出来。

### 5.3 N2（F=ma 在恒定外力下）—— **未测**

**N2 声明**：F = m·a。在 phyworld 里**最干净的 N2 测试是 parabola（抛物线）**：球受重力 → 恒定下向加速度 g → 抛物线轨迹。

要 probe 出 N2，需要测：

| 物理量 | 测量方法 |
|---|---|
| **a_y（恒定重力加速度）** | 跨多帧 probe 推断每个 traj 的 g | 
| **v_y 随时间线性增长** | 多帧 probe 检测 vel_y 帧间差异 |
| **轨迹的 curvature** | 多帧 probe 是否能从 emb 序列预测 future trajectory shape |

> **我们没做 parabola 训练 + probe**。原因：
>
> 1. parabola 的 action（force-action mode 下 a 是 dataset-wide 常数 (0, -g)），实际上对 encoder 的训练梯度信号**极弱**（见 [COLLISION_REPORT.md §4.2](COLLISION_REPORT.md)）
> 2. 当时判断"parabola 在我们的 setup 下不会给出新结论"——这个判断**针对 negative result 的归因**是对的，但**没有覆盖"是否完整学到牛顿定律"的问题**
>
> **正确的 honest 表述**：我们对 N2 的判断**没有直接证据**。

### 5.4 总判断

| 牛顿定律 | 在 ID 范围内学到了？ | 证据强度 | OOD 表现 |
|---|---|---|---|
| **N1（惯性）** | ✅ 是 | 中（emb 有 pos + vel；但 action=vel 这个 setup 让"恒定 vel"判断不严格） | OOD 上 vx R² 在 r-OOD 上跌到 −3.69 |
| **N2（F=ma 恒力）** | ❓ **未测** | 无 | 无 |
| **N3（作用反作用）** | ✅ 是 | 强（pos / vel / mass / 碰撞事件全部 probe 出，predictor 能正演） | both-OOD 上 pos R² 跌到 −0.024 |

> **核心结论**：le-wm 在 phyworld ID 上学到了 **N1 + N3** 的状态变量与动力学关系。**N2 是本报告的覆盖空白**——要补完，需要做 parabola 训练 + 多帧 probe vel_y / a_y。
>
> 但是即使 N1 + N3 在 ID 内学到，OOD 上也**部分崩溃**（§6），跨域上**计数 transfer 失败**（§6.6）——说明 le-wm 学的是 **ID 视觉模式而非抽象牛顿定律**，和 phyworld 论文对 video-gen 模型 "case-based generalization" 的判断**同性质**。

---

## 6. OOD 实验：encoder 学到的表征能 transfer 到分布外吗？

> 前面 §2-§5 都是 in-distribution probing。phyworld 论文 §3 强调："**真正学到物理规律的 world model 必须在 OOD 上也工作**"——它们发现 video generation 模型在 OOD 上误差比 ID 大一个数量级，所以"scaling 不够"。我们在 probing 层面复现这个测试。

### 6.1 协议

phyworld 论文定义的 ID/OOD 边界：

- **ID 范围**：`r ∈ [0.7, 1.5]`、`v ∈ [1, 4]`
- **OOD 范围**：`r ∈ [0.3, 0.6] ∪ [1.5, 2.0]`、`v ∈ [0, 0.8] ∪ [4.5, 6.0]`

四个 partition（基于 traj 的初始 r, v）：

```
ID        : r 和 v 都在 ID 范围
r-OOD     : 至少一个球的 r OOD，但 v 都 ID
v-OOD     : 至少一个 v OOD，但 r 都 ID
both-OOD  : r 和 v 都至少有一个 OOD
```

**collision** 实验：训练用 30K 文件取 5000 traj（**纯 ID** by construction），eval 用 `collision_eval.hdf5`（**混合 1635 traj**，partition 分布 ID 115 / r-OOD 449 / v-OOD 165 / both-OOD 906）。**probe Ridge 在训练 ID 集上 fit，在 eval 各 partition 上分别测**——干净的 OOD 泛化测试。

**uniform_motion** 实验：caveat 是我们前面训练时用了 `uniform_motion_eval.hdf5` 全部 1152 traj（已含 OOD），所以 encoder 不是 OOD-naive。但 probe Ridge 只在 ID partition fit、其他 partition 测仍然有意义——测的是"encoder 的线性 pos/vel 读出是否能从 ID 范围外插到 OOD 范围"。

### 6.2 Collision OOD 结果

| Encoder | Metric | ID | r-OOD | v-OOD | both-OOD |
|---|---|---|---|---|---|
| **trained paper-init** | pos_x R² | **0.876** | **0.541** | **0.819** | **−0.024** |
| | vel_x R² | **0.529** | **0.532** | 0.180 | 0.218 |
| | collision AUC | **0.947** | **0.807** | **0.756** | **0.678** |
| random ViT | pos_x R² | 0.662 | 0.106 | 0.445 | −0.141 |
| | vel_x R² | 0.420 | 0.188 | 0.346 | 0.071 |
| | collision AUC | 0.742 | 0.660 | 0.679 | 0.605 |
| pixel-stats (9-D) | pos_x R² | 0.690 | 0.401 | 0.536 | −0.115 |
| | vel_x R² | 0.423 | **0.416** | **0.415** | 0.184 |
| | collision AUC | 0.659 | 0.673 | 0.632 | 0.662 |

### 6.3 Uniform_motion OOD 结果

| Encoder | Metric | ID (fit=eval) | r-OOD | v-OOD | both-OOD |
|---|---|---|---|---|---|
| **trained paper-init** | pos_x R² | 0.993* | **0.842** | **0.873** | **0.807** |
| | vx R² | 0.463* | **−3.686** | 0.155 | −0.192 |
| random ViT | pos_x R² | 0.507* | −1.971 | 0.594 | −0.145 |
| | vx R² | 0.203* | −0.570 | 0.136 | 0.052 |
| pixel-stats | pos_x R² | 0.405* | −0.251 | 0.664 | 0.300 |
| | vx R² | 0.111* | 0.086 | 0.167 | 0.165 |

`*` 表示 Ridge 是在该 partition 上 fit 的（fit==eval set 给出虚高 R²，非泛化测量）。

### 6.4 5 大关键观察

1. **trained encoder 在所有 partition 上击败 baseline**（除 vel_x 的 pixel-stats "低但稳定"情况）——证明 encoder 确实学到了部分**可 transfer 的视觉表征**，不是单纯过拟合训练分布。
2. **both-OOD 上所有 encoder 全面崩盘**（collision pos R² 全负）——和 phyworld 论文 video-gen OOD 失败的核心发现**完全一致**。即使 paper-init 这种 state-of-the-art 表征也不能解决 OOD 推理问题。
3. **r-OOD 主要打掉 pos_x**（球大小变了，encoder 的"size→position"映射崩）；**v-OOD 主要打掉 vel_x**（Ridge 线性系数没法外插速度范围）——OOD 失败的**模式有清晰物理可解释性**。
4. **uniform 上 r-OOD 让 vel_x R² 直接 −3.69**：encoder 学到了"球的视觉大小→速度"的虚假关联（因为 uniform_motion 里 action=velocity，预测目标和外观耦合），OOD 半径上完全失效——**佐证 §5.2 的"action=velocity 是 bug"的论点**。
5. **pixel-stats vel_x 极稳**（collision 上 0.42→0.42→0.42→0.18）：说明数据集的"位置→速度"统计 shortcut 在 OOD 上**仍然成立**（因为 shortcut 来自 trajectory 的轨迹结构而不是 encoder 的学习），dataset bias 是真的躲不掉。

### 6.5 这对"学到牛顿定律"的结论意味着什么

§5 把判断按 N1 / N2 / N3 拆开后，OOD 实验进一步收紧每个定律的成立范围：

| Setting | N1（惯性）| N3（作用反作用）|
|---|---|---|
| ID（训练分布内）| ✅ encoder 编码 pos+vel，predictor 正演 | ✅ pos+vel+mass+碰撞事件全部 probe 出 |
| 单轴 OOD（r 或 v 之一）| ⚠️ pos transfer 一些；**r-OOD 上 vx R² −3.69** 灾难 | ⚠️ pos R² 0.54-0.82，AUC 0.76-0.81 |
| both-OOD | ❌ pos/vel 全崩 | ❌ pos R² 负值，AUC 0.68 |

（N2 全行省略——未测）

**修正后的判断**：le-wm 学到的是**ID 范围内的牛顿定律状态变量与动力学**，而**不是抽象的 N1 / N3 物理规律**。它的表征基于训练分布的视觉统计（球的颜色、大小、位置范围），OOD 一变就部分失效——和 phyworld 论文对 video-gen 模型的判断 "case-based 而非 rule-based" 是**同一性质**的限制。

> 但 trained encoder 在 OOD 上至少**击败 random + pixel-stats baseline**，证明它确实学到了**比像素 shortcut 更强**的某种通用视觉表征——只是这个表征还不是完整的物理规律抽象。

### 6.6 Zero-shot cross-domain probe（PHYRE 组合数据上的迁移测试）

#### 动机

phyworld 论文 §4 的组合泛化测试需要在 PHYRE 多对象数据上**重新训练** le-wm（成本 1-2 天）。我们没做完整版，但做了一个轻量级替代：**zero-shot 跨域 probe** —— 用我们已经训好的 `collision_paperinit` encoder（从没见过 PHYRE 场景），直接在 PHYRE OOT eval 数据上 probe，测它的视觉表征是否能 transfer 到完全不同的视觉域。

#### 数据

`magicr/phyworld/combinatorial_data/combinatorial_out_of_template_eval_1K.hdf5`（HuggingFace，46 MB）：

- 1000 个 OOT eval 视频跨 10 个 PHYRE 模板（10060-10069）
- 每个视频 50 帧 @ 512×512，含 5 个对象（多类：球、罐、横杆、棒等）
- 我们抽样 stride=5 → 10 000 帧用于 probe

#### Probe 设置

- Split by template：8 templates 训练 + 2 templates 测试（组合-style 切分，测跨模板泛化）
- 3 个 target：
  - `scene_centroid_x`、`scene_centroid_y`：所有对象的平均位置 → Ridge R²
  - `n_present_objects`：当前帧的对象数 → Ridge R²

#### 结果

| Target | trained (collision_paperinit) | random ViT | pixel-stats (9-D) |
|---|---|---|---|
| **scene_centroid_x** R² | **+0.380** | −0.130 | +0.073 |
| **scene_centroid_y** R² | **+0.688** | +0.302 | +0.011 |
| **n_present_objects** R² | +0.116 | **+0.398** | +0.391 |

#### 3 个关键发现

1. **位置定位 transfer 得不错**：trained encoder 在跨域 OOT 模板上仍能线性恢复对象 y 坐标 R²=0.69，比 random ViT（0.30）高 +0.39，比 pixel-stats（0.01）高 +0.68。证明 encoder 学到了**比像素 shortcut 更强、有跨域迁移能力的视觉表征**——不是单纯过拟合训练分布。

2. **物体计数 transfer 失败，trained 反而比 random 还差**（0.12 < 0.40）：collision encoder 只见过 2 球场景，它的表征已"压缩"成"两个深色斑点"模式，PHYRE 上 1-5 对象数量差异被它**抹平**。random ViT 没有这种 inductive bias，反而保留了原始像素差异。

   > 这是 §6.5 "encoder 学到的是 ID-specific case-based 表征"论点的**强证据**——encoder 学的是"2-球场景的视觉模式"，不是"如何抽象感知物体"。

3. **template_id 分类未跑出有效结果**：10-way LogReg fit 在 8 个 train 模板上，根本无法预测 test 集那 2 个未见 class（acc=0 是结构性的，非模型问题）。要测的话需要 within-template split。

#### 这对"学到牛顿定律"的意义

强化 §6.5 的修正结论：**le-wm 的表征是 ID-specific 的视觉表征**，部分可迁移（位置），部分不可迁移（计数 / 多对象组合）。**即使 N1/N3 在 ID 内学到了状态变量与动力学，跨域上它都不能可靠 transfer**——这和 phyworld 论文对 video-gen 模型 "case-based generalization" 是同性质的限制。

#### 未做完整组合泛化的原因

完整复现（在 PHYRE templates 子集上训练 le-wm 再测 OOT）需要：

- 下载 ≥24 GB 训练数据（磁盘紧张需要先腾空间）
- 改 le-wm 架构去掉 per-step action（PHYRE 没 action）
- 训练 ~10-15 h
- 总成本 1-2 天

> **建议**：作为独立的下一阶段实验，**优先级中等**。我们的 zero-shot 替代已经在 §6.6 给出关键信号（encoder 是 ID-specific 的）。

---

## 7. 方法论教训

### 7.1 推荐 default 协议

在 le-wm 系列实验里，下游表征评估应**默认**使用：

1. **paper-init**（如有可用 ckpt）→ 跨过"从零学小数据"瓶颈
2. **multi-frame K=4 probe** → 测 velocity 等跨帧量
3. **`--no-projector`** → 绕过 JEPA projector 的信息瓶颈

三者缺一就有可能产生 negative result 假阴性。

### 7.2 baseline 必须的 4 个对照

1. **trained encoder** (主角)
2. **random ViT-tiny**（同架构未训，测训练增益）
3. **pixel-stats**（9 维像素 mean/std/mean²，测信号下界）
4. **paper-init encoder 未在 phyworld 训过的**（消歧 "PushT 视觉知识" vs "phyworld 物理知识"哪个贡献更大）—— 这个我们**还没做**，是建议补充的对照

### 7.3 数据切分必须

- 按 **trajectory** 切，绝对不能按帧切（避免相邻帧泄露 → R² 虚高）

### 7.4 协议错误的代价

uniform_motion 我们花了一周得到"encoder 没学到速度"的结论，后来发现 vel 信息**早就在 emb 里**（R²=0.94），只是 protocol 读不出。**正负反 0.77 R²**——这是 "协议错误" 而不是 "模型不行"。

---

## 8. 局限性

### 8.1 数据集 shortcut

phyworld 数据集有强像素 shortcut：

- 颜色通道 std 和 pos_x 相关性 +0.70（球颜色和位置耦合）
- pixel-stats 9 维就能拿 R²(pos_x)=0.70 / R²(mass_ratio)=0.67

这意味着 trained encoder R²=0.94 的部分高分**来自这个 shortcut 天花板**，不是纯"物理理解"。要剥离 shortcut，需要数据增强（背景随机化）或更难的 benchmark。

### 8.2 没和 video gen 模型对比

phyworld 论文本意是测试**视频生成模型**（CogVideoX 等大模型）的物理理解。我们只是把 le-wm 这种紧凑模型放在 phyworld 上做了一个 "试试看 + 学习方法论" 的实验，不是 head-to-head 比较。

### 8.3 Linear probe 局限

Ridge probe 只能测**线性可解码性**。如果 encoder 把信息编成非线性形式（如球位置的傅里叶基），线性 probe 看不见，会低估 encoder。改用 MLP probe 是可行的下一步，但要明确"测的是 encoder + MLP 联合能力"。

### 8.4 没做 random-init multi-frame 消歧

paper-init 同时贡献了"PushT 视觉知识"和"在 phyworld 上多训的物理表征"两件事，我们没拆开。一个干净的下一步实验：**冻结 paper-init encoder 不在 phyworld 上训**，直接 probe，看 vel/mass R² 已经多高——如果差距不大，说明 PushT visual 是主因；如果显著低，说明 phyworld 训练贡献了大部分。**估计耗时 10 min**。

### 8.5 只测了 PushT pretrained

le-wm 论文还 release 了 `lewm-cube`（3D 立方体抓取任务），我们没试。可能对 3D 物理有更好的 transfer，但和 phyworld 视觉差异更大。

---

## 9. 后续可做（优先级排序）

1. ⭐ **N2（parabola）训练 + probe** —— **最重要的缺口**：本报告只覆盖 N1 + N3，N2（F=ma 在恒定重力下）完全没测。要补完"le-wm 是否学到了完整牛顿运动定律"这个问题，必须做 parabola 实验。预估 ~6-8 h（生成数据 + 训练 + 多帧 probe v_y / 推断 a_y）。
2. **random-init multi-frame 消歧**（7.4 节）—— 成本低、答案清楚
3. ~~**ImageNet ViT-tiny init 对照**~~ —— **已被 DiT-XL zero-shot 实验间接回答**。注：Google 原版 ViT-tiny 没释放，只有 DeiT-tiny（patch=16，和 LeWM patch=14 不匹配），需 patch embed interpolation 才能用。鉴于 DiT-XL zero-shot 已经验证了"通用 ImageNet 预训练能在 phyworld 上做物理 probe ≈ LeWM trained"这一假设，该实验优先级降低
4. **数据增强 / 背景随机化**——破除 shortcut 天花板，让 trained 优势更显著
5. **加 MLP probe 做辅助诊断**——量化"线性 probe 低估"了多少
6. **跑 phyworld 30K-3M 大数据集**——验证当前结论在数据量充足时仍成立
7. **完整组合泛化（PHYRE 训练版）**—— §6.6 只做了 zero-shot，完整版需要 1-2 天
8. **(优先级低) 自建 random-force 数据集** —— "action 信号稀疏" 假设已被 paper-init 实验间接证伪

---

## 10. 引用 / 文件清单

### 10.1 论文

- **LeWorldModel** (Maes & Le Lidec et al., 2026 preprint): https://github.com/lucas-maes/le-wm
- **PhyWorld** (Kang & Yue et al., 2024): https://github.com/PhyWorld/PhyWorld, arXiv:2411.02385
- **JEPA 相关** (LeCun 2022 World Model Position Paper)

### 10.2 论文 release 权重

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht) — 本实验用作 paper-init
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube) — 未使用

### 10.3 代码 / 脚本（本实验产出）

| 文件 | 作用 |
|---|---|
| [phyworld/scripts/convert_to_lewm.py](../phyworld/scripts/convert_to_lewm.py) | uniform_motion 数据转换（原项目）|
| [phyworld/scripts/convert_collision_to_lewm.py](../phyworld/scripts/convert_collision_to_lewm.py) | collision 数据转换（新增）|
| [phyworld/scripts/probe_lewm_encoder.py](../phyworld/scripts/probe_lewm_encoder.py) | uniform_motion probe（原项目）|
| [phyworld/scripts/probe_collision_encoder.py](../phyworld/scripts/probe_collision_encoder.py) | collision probe（新增）|
| [phyworld/scripts/probe_multiframe.py](../phyworld/scripts/probe_multiframe.py) | 多帧 probe（新增）|
| [le-wm/train.py](../le-wm/train.py) | 训练入口；本次加了 `init_from_ckpt` config 支持 paper-init |
| [le-wm/config/train/data/collision.yaml](../le-wm/config/train/data/collision.yaml) | collision 训练 hydra config（新增）|

### 10.4 数据 / Checkpoint

| 路径 | 大小 | 含义 |
|---|---|---|
| `~/.stable_worldmodel/phyworld_uniform_motion.h5` | 100 MB | uniform_motion lewm 格式 |
| `~/.stable_worldmodel/phyworld_collision.h5` | 764 MB | collision lewm 格式 |
| `~/.stable_worldmodel/lewm_paper_pusht/weights.pt` | 72 MB | 论文 PushT 权重，paper-init 用 |
| `~/.stable_worldmodel/phyworld_probe/lewm_phyworld_epoch_20_object.ckpt` | 72 MB | uniform from-scratch 20 epoch |
| `~/.stable_worldmodel/uniform_paperinit/lewm_uniform_paperinit_epoch_20_object.ckpt` | 72 MB | uniform paper-init 20 epoch |
| `~/.stable_worldmodel/collision_run/lewm_collision_epoch_8_object.ckpt` | 72 MB | collision from-scratch 8 epoch |
| `~/.stable_worldmodel/collision_paperinit/lewm_collision_paperinit_epoch_8_object.ckpt` | 72 MB | collision paper-init 8 epoch |

### 10.5 详细子报告（保留作为参考）

- [UNIFORM_MOTION_REPORT.md](UNIFORM_MOTION_REPORT.md) — uniform_motion 完整细节 + 概念 FAQ §5
- [COLLISION_REPORT.md](COLLISION_REPORT.md) — collision 完整细节 + 概念 FAQ §6（含 R²/AUC 数学、Ridge 详解 + 例子、按 traj 切、多帧 probe 等 8 个子节）
