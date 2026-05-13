---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 22px; padding: 40px 60px; }
  h1 { color: #164A80; }
  h2 { color: #164A80; border-bottom: 2px solid #164A80; padding-bottom: 6px; }
  table { font-size: 18px; margin: 0 auto; }
  th { background: #164A80; color: white; }
  tr:nth-child(even) { background: #EEEEEE; }
  .highlight { background: #FFF3CD; color: #C0392B; font-weight: bold; }
  .accent { color: #C0392B; font-weight: bold; }
  .green { color: #1F7744; font-weight: bold; }
  code { background: #F5F5F5; padding: 2px 6px; border-radius: 3px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# le-wm 能否学到牛顿第二定律？

### 在 phyworld 物理仿真上的 probing 实验报告

<br>

2026-05-10 · haochenluo02@gmail.com · 协助：Claude

---

## 提纲

1. **研究问题**：什么叫"学到牛二定律 F = ma"
2. **实验设计**：phyworld 数据 + le-wm 架构 + linear probe 协议
3. **初版结果**：trained ≈ random ≈ pixel-stats（看似 negative）
4. **三个 confound 联合造成的假阴性**
5. **修正后的结果**：encoder 实际学到了所有物理量
6. **关于牛二定律的具体答复**
7. **方法论教训 + 局限 + 后续**

---

## 研究问题：什么叫"学到牛二定律"

**牛顿第二定律 F = m · a 串起 3 个物理量 + 2 个状态变量**：

- **m**　 质量 — 每物体一个常数
- **a**　 加速度 — 每物体每时刻 2D 向量
- **F**　 合外力 — 每物体每时刻 2D 向量
- **pos**　位置 — 对 vel 积分得到
- **vel**　速度 — 对 a 积分得到

<br>

**le-wm 是 JEPA 自监督模型，只用 pixel 训，没 label。怎么知道学到了？**

→ 用 **linear probe**：冻结 encoder，只训一个 Ridge 从 emb 预测物理量。

R² 高 → encoder 把这个量**线性可读地**编进了 emb 里。

---

## 牛二定律 → probe target 映射

| 物理量 | 测量方式 | 在 le-wm 哪里 |
|---|---|---|
| pos | 单帧 probe Ridge → (x, y) | encoder emb |
| vel | 多帧 K=4 probe Ridge → vx | encoder emb（跨帧）|
| m | collision 上 probe mass_ratio = m₁/m₂ | encoder emb |
| F (碰撞冲量) | collision_event 二分类 LogReg → AUC | encoder emb + action |
| a | action 直接读 / 多帧二阶差分 | action 输入 |
| F = ma 动力学 | pred_loss (emb 空间 MSE) | predictor |

---

## le-wm 架构（JEPA 风格 World Model）

```
                                  ┌── action_encoder ──┐
   图 (B,T,3,224,224)                          action (B,T, 2 or 4)
        │                                            │
        ▼                                            ▼
   ViT-tiny encoder                            act_emb (B,T,192)
   (5.5 M params, patch=14)                          │
        │                                            │
   cls_token (B,T,192)                               │
        │                                            │
   projector (MLP)                                   │
        │                                            │
   emb (B,T,192) ───────┬── tgt_emb[:,1:]──┐         │
                         │                  └────────┤
                         ▼                           ▼
                  AR predictor (6-layer causal, AdaLN-zero)
                         │
                         ▼
                    pred_emb (B,T,192)

   Loss = ||pred_emb − tgt_emb||²  +  0.09 · SIGReg(emb)
              ↑                              ↑
         JEPA 预测 loss               推 emb 分布 → N(0, I)
```

总参数 ~10 M，ViT-tiny 主导。**SIGReg 是 le-wm 核心创新**（单 loss 项防 collapse）。

---

## phyworld 数据集（被动物理观测）

| | uniform_motion (匀速直线) | collision (弹性碰撞) |
|---|---|---|
| 球数 | 1 球 | 2 球 |
| 物理 | 0 力 | 弹性碰撞 |
| trajectories | 1 152 | 5 000 |
| 帧数 | 36 864 | 160 000 |
| action 类型 | velocity-action: a = vel | force-action: a = 2 阶差分 |
| action 维度 | 2 | 4（2 维恒为 0）|

<br>

**关键设计选择：collision 用加速度作 action（不用速度）。**

若 action=velocity，predictor 直接拿到 vel → encoder 不需要编码 vel → probe 给"误导性低 R²"，分不清"学不会"还是"有意不学"。uniform_motion 的关键 bug 就在于此。

---

## Probe 协议

- **数据切分**　　按 trajectory 80/20 切（不按帧！避免相邻帧泄漏）
- **线性模型**　　Ridge α=1.0（回归）／LogReg class_weight=balanced（分类）
- **评估指标**　　R² (上限 1.0) ／ AUC (上限 1.0)
- **K=1 单帧**　　emb[t] → Ridge → target（192-D 输入）
- **K=4 多帧**　　concat 4 帧 emb → Ridge（768-D 输入），跨帧线性差分恢复 velocity
- **Projector 开关**　默认 with-proj；`--no-projector` 直接读 cls token，绕过 JEPA 信息瓶颈
- **3 个 baseline**　trained ／ random ViT ／ pixel-stats (9-D)

---

## 实验矩阵：8 个 setup

| 维度 | 可选项 | 说明 |
|---|---|---|
| 数据集 | uniform_motion / collision | 两种物理 |
| 初始化 | from-scratch / paper-init | paper-init 用 lewm-pusht ckpt |
| Probe | K=1 / K=4 | 单帧 vs 4 帧 concat |
| Projector | with-proj / no-proj | no-proj 推荐 |

<br>

总共 2 × 2 × 2 × 2 = 16 个 cell。我们跑了关键的 **8 个 cell** 完成核心对比。

---

## 初版结果：trained ≈ random ≈ pixel-stats

**uniform_motion**（from-scratch + 单帧 + with-projector）

| Encoder | x_pos | vx |
|---|---|---|
| trained | 0.732 | **0.166** |
| random ViT | 0.582 | 0.227 |
| pixel-stats | (合并) | **0.256** |

**collision**（from-scratch + 单帧 + with-projector）

| Encoder | pos_x | vel_x | mass | AUC |
|---|---|---|---|---|
| trained | 0.708 | 0.473 | 0.719 | 0.741 |
| random ViT | 0.649 | 0.464 | 0.666 | 0.763 |
| pixel-stats | 0.700 | 0.470 | 0.666 | 0.667 |

<br>

<span class="accent">看似结论：encoder 几乎没学到东西。uniform vx 上 trained < random < pixel-stats。</span>

---

## Plot Twist：3 个 confound 联合假阴性

**原 negative result 不是模型问题，是协议 + 初始化的 3 个 confound 联合放大出来的。**

- **Confound A：projector 信息瓶颈**
  JEPA loss 在 projector 输出空间算，projector 会压掉对预测无用的维度

- **Confound B：单帧 probe 协议盲区**
  速度本质是跨帧差分，单帧 emb 内部没法做"减法"

- **Confound C：random init + 小数据学不动**
  5.5M ViT-tiny + 36k~160k 帧 = 比标准 ViT 训练少 100-10000 倍数据

<br>

下面分别看每个 confound 的实测影响。

---

## Confound A：projector 压信息

JEPA loss = ‖pred(z[t+1]) − z[t+1]‖²，loss 在 projector 输出 z 上算。projector 被训练优化预测下一帧 → 主动压掉对预测无用的维度（信息瓶颈）。

下游 probe 要的是"物理量信息"，不是"预测足够性" → **必须绕过 projector**。

**实测**（collision from-scratch）：

| Target | with-projector | --no-projector | Δ |
|---|---|---|---|
| pos_x R² | 0.708 | 0.726 | +0.018 |
| collision AUC | 0.741 | **0.808** | **+0.067** |

collision_event 上 +0.067 AUC 最显著：碰撞瞬间对"预测下一帧"无用（时间极短），projector 把它压掉了。

---

## Confound B：单帧 probe 协议盲区

**单帧 probe** = emb[t] 一个 192-D 向量给 Ridge，要求"vel 必须线性编码进单个向量"。但 vel = (pos[t] − pos[t-1]) / Δt 本质是**跨帧差分**——一张静态照片看不出球速度。

**多帧 probe** = concat 4 帧 emb，让 Ridge 跨时间步学线性差分组合。

**实测**：

| Setup | K=1 vel_x R² | K=4 vel_x R² | Δ |
|---|---|---|---|
| **uniform paper-init** | **0.232** | **0.939** | **+0.707** |
| collision paper-init | 0.620 | 0.883 | +0.263 |
| collision from-scratch | 0.487 | 0.594 | +0.107 |

<br>

<span class="accent">uniform 上 +0.71 跳跃 → vel 信息确实在 emb 里，单帧 protocol 读不出。</span>

---

## Confound C：random init + 小数据学不动

- ViT-tiny: **5.5 M 参数**
- 标准 ViT 训练：ImageNet-21k (14M 图) 或 JFT-300M (300M 图)
- 我们 collision: 160k 帧；uniform: 36k 帧 — **少 100-10000 倍**
- lewm-pusht paper init：PushT 上预训的 encoder，给 phyworld 做 fine-tune

**实测**（collision，K=1 no-projector）：

| Init | pos_x | vel_x | mass | AUC |
|---|---|---|---|---|
| from-scratch | 0.726 | 0.487 | 0.711 | 0.808 |
| **paper-init** | **0.863** | **0.620** | **0.826** | **0.952** |
| Δ | +0.137 | +0.133 | +0.116 | +0.144 |

<br>

所有 target 同步 **+0.12 ~ +0.14**——paper-init 是 negative result 主因之一。

---

## 修正后结果：4-way 对比（uniform_motion）

| Setup | x_pos R² | y_pos R² | vx R² |
|---|---|---|---|
| from-scratch + K=1 + with-proj（原 protocol） | 0.732 | −0.137 | 0.166 |
| paper-init + K=1 + no-proj | 0.942 | 0.859 | 0.232 |
| **paper-init + K=4 + no-proj  ★** | **0.964** | **0.964** | **0.939** |

<br>

<span class="accent">vx R² 从 0.166 → 0.939（+0.77）</span>

encoder 几乎完美编码 1D 物理状态。原版"trained vx 反而比 random 还差"是 3 confound 联合假阴性。

---

## 修正后结果：4-way 对比（collision）

| Setup | pos_x | vel_x | mass | AUC |
|---|---|---|---|---|
| from-scratch + K=1 + with-proj（原 protocol） | 0.708 | 0.473 | 0.719 | 0.741 |
| from-scratch + K=1 + no-proj | 0.726 | 0.487 | 0.711 | 0.808 |
| from-scratch + K=4 + no-proj | 0.814 | 0.594 | — | — |
| paper-init + K=1 + no-proj | 0.863 | 0.620 | 0.826 | 0.952 |
| **paper-init + K=4 + no-proj  ★** | **0.911** | **0.883** | — | — |

<br>

<span class="accent">★ encoder 真实能力：pos 0.91 / vel 0.88 / mass 0.83 / AUC 0.95</span>

从弱单帧 0.47 一路打到强多帧 0.88，confound 修复带来的跃迁明显。

---

## 关于牛二定律 F = ma 的具体答复

| 组件 | 在哪 | 怎么验证 | 数值 |
|---|---|---|---|
| pos | encoder emb | K=1 probe | R² = 0.86 ~ 0.94 |
| vel | encoder emb（跨帧）| K=4 multi-frame | R² = 0.88 ~ 0.94 |
| m（质量）| encoder emb | K=1 probe | R² = 0.83 |
| F（碰撞冲量）| encoder emb + action | 二分类 LogReg | AUC = 0.95 |
| a（加速度）| action 输入 + emb | 直接读 / 多帧二阶差分 | 完全可得 |
| F = ma 动力学 | predictor | pred_loss | 0.017（emb-space MSE）|

<br>

<span class="green">结论：le-wm 学到了 F=ma 所需的全部物理量 + 隐式动力学。</span>

---

## 但是…… 这只是 In-Distribution 结果

到此 §1-§5 全是 **训练分布内** 的 probing。phyworld 论文 §3 强调："**真正的 world model 必须在 OOD 上也工作**"——他们发现 video-gen 模型 OOD 误差比 ID 大一个数量级。

**我们也得 OOD probe 一下，看 le-wm 学到的是真物理规律还是 ID-specific 视觉 shortcut。**

下面 3 张片：OOD 设置 + collision/uniform 结果 + 对结论的修正。

---

## OOD 设置（沿用 phyworld 论文协议）

phyworld 论文定义的 ID/OOD 边界：

- **ID 范围**：`r ∈ [0.7, 1.5]`、`v ∈ [1, 4]`
- **OOD 范围**：`r ∈ [0.3, 0.6] ∪ [1.5, 2.0]`、`v ∈ [0, 0.8] ∪ [4.5, 6.0]`

四个 partition：**ID** / **r-OOD** / **v-OOD** / **both-OOD**

**协议**：

- collision：训练用 30K 的 5000 ID-only traj；eval 用 `collision_eval.hdf5` 混合 1635 traj
- probe Ridge **只在 ID frames 上 fit**，在 eval 的 4 个 partition 上分别测 R²
- 测试 trained encoder 的线性表征是否能从训练分布外插

---

## OOD 结果：collision

| Encoder | Metric | ID | r-OOD | v-OOD | **both-OOD** |
|---|---|---|---|---|---|
| **trained paper-init** | pos_x R² | **0.876** | **0.541** | **0.819** | **−0.024** |
| | vel_x R² | **0.529** | **0.532** | 0.180 | 0.218 |
| | coll AUC | **0.947** | **0.807** | **0.756** | **0.678** |
| random ViT | pos_x R² | 0.662 | 0.106 | 0.445 | −0.141 |
| | coll AUC | 0.742 | 0.660 | 0.679 | 0.605 |
| pixel-stats | pos_x R² | 0.690 | 0.401 | 0.536 | −0.115 |

<span class="accent">trained encoder 在每个 partition 都击败 baseline。但 both-OOD 上 pos R² 全部跌到 0 以下——比猜均值还差。</span>

---

## OOD 结果：uniform_motion

| Encoder | Metric | r-OOD | v-OOD | both-OOD |
|---|---|---|---|---|
| **trained paper-init** | pos_x R² | **0.842** | **0.873** | **0.807** |
| | vx R² | **−3.686** | 0.155 | −0.192 |
| random ViT | pos_x R² | −1.971 | 0.594 | −0.145 |
| | vx R² | −0.570 | 0.136 | 0.052 |
| pixel-stats | pos_x R² | −0.251 | 0.664 | 0.300 |

uniform 上 pos transfer 不错（trained 0.81-0.87），但 **vx R² 在 r-OOD 上 −3.69 灾难性失败**——encoder 学到了"球的视觉大小 → 速度"的虚假关联（因为 action=velocity，外观和目标耦合），换个球大小就崩。

---

## OOD 失败的物理可解释模式

- **r-OOD 主要打掉 pos_x**：球大小变了 → encoder 的"视觉 size → 位置"映射失效
- **v-OOD 主要打掉 vel_x**：Ridge 在 v∈[1,4] 上拟合的线性系数，外插到 v∈[4.5,6] 上直接崩
- **both-OOD**：两个 confound 叠加，全面崩溃

<br>

**和 phyworld 论文 video-gen 模型 OOD 失败的核心结论一致**：

> "video generation models fail to abstract general physical rules and instead exhibit case-based generalization behavior"

我们的 probe 上也观察到同样的"基于 case 的"行为——encoder 学到的是**训练分布的视觉统计**，不是抽象的 F=ma 规则。

---

## 修正后的"学到牛二定律"答复

| Setting | 是否学到 F=ma 所需物理量？ |
|---|---|
| ID（训练分布内）| ✅ 完美编码 pos/vel/mass，predictor 隐式实现 F=ma |
| 单轴 OOD（r 或 v 之一）| ⚠️ 部分成立——pos/vel R² 大幅降但 trained 仍胜 baseline |
| **both-OOD** | ❌ 编码崩溃，pos R² 负值，**没学到通用 F=ma 抽象** |

<br>

<span class="green">**最终判断**：le-wm 学到了**ID 分布内**的物理状态可读表征，但**不是抽象的 F=ma 规律**。它的表征基于训练分布的视觉统计，OOD 一变就部分失效——和 phyworld 论文对 video-gen 模型的判断"case-based"是**同性质**的限制。</span>

<br>

但 trained encoder **在 OOD 上至少打败 random+pixel-stats**，证明它学到了**比像素 shortcut 更强**的视觉表征。只是这表征还不是"物理规律"。

---

## Bonus 实验：Zero-shot cross-domain probe（PHYRE 跨域测试）

phyworld 论文 §4 的 combinatorial generalization 完整复现需要 1-2 天。我们做了**轻量级替代**：

- 从 HuggingFace 下载 PHYRE OOT eval 1K（46 MB）
- 用 **collision_paperinit encoder**（从没见过 PHYRE 场景）**zero-shot** probe PHYRE 多对象场景
- Split by template：8 train / 2 test

测的不是真正组合泛化（需要在 PHYRE 训练），而是：**collision encoder 的视觉表征能否 transfer 到完全不同的视觉域？**

---

## PHYRE 跨域结果

| Target | trained (collision_paperinit) | random ViT | pixel-stats |
|---|---|---|---|
| **scene_centroid_x R²** | **+0.380** | −0.130 | +0.073 |
| **scene_centroid_y R²** | **+0.688** | +0.302 | +0.011 |
| **n_present_objects R²** | +0.116 | **+0.398** | +0.391 |

<br>

<span class="accent">关键发现</span>：

1. **位置定位 transfer 得不错**：centroid_y R² 0.69 vs random 0.30 —— encoder 学到了**真有跨域能力**的视觉表征
2. **物体计数 transfer 失败**：trained R²=0.12 比 random 0.40 还差—— encoder 把多对象场景"压缩"成"2 球"模式，**count 信息丢了**

> 这是 §5/§6 "encoder 学的是 ID-specific case-based 视觉表征" 论点的**强证据**——它学的是"2-球场景的视觉模式"，不是"如何抽象感知物体"。

---

## 结论（最终版）

- **主问题答复**　le-wm 学到了 **ID 范围内**的 F=ma 所需物理量；
  **OOD 上部分失效**（pos R² 转负），**跨域上 count 失败**（trained < random）
  和 phyworld 对 video-gen 模型 **"case-based generalization"** 的判断**同性质**

- **方法论教训**　default protocol = `paper-init` + `multi-frame K=4` + `--no-projector`
  3 个 confound 任一缺失都会产生假阴性 negative result

- **未做实验**　phyworld §4 完整 combinatorial generalization（PHYRE 训练版）
  工程量 1-2 天 + 24 GB+ 训练数据 + 改 le-wm 架构，建议下阶段独立立项

---

## 局限性 + 后续可做

**局限性**

- phyworld 强像素 shortcut：颜色通道 std × pos_x 相关性 +0.70 / pixel-stats 9-D 拿 R²=0.70
- 没和 video gen 模型（CogVideoX 等）head-to-head 对比
- Linear probe 只测线性可读性
- 没拆开 paper-init 的"PushT 视觉知识" vs "phyworld 物理表征"贡献

**后续可做（按优先级）**

- **[高]** random-init multi-frame 消歧实验（10 min 成本）
- **[中]** ImageNet ViT-tiny init 对照（验证是否 paper-init 不必）
- **[中]** 数据增强 / 背景随机化（破 shortcut 天花板）
- **[低]** 自建 random-force 数据集（paper-init 实验已间接证伪相关假设）

---

## 引用 / 资源

- **LeWorldModel** — Maes et al. 2026 preprint · github.com/lucas-maes/le-wm
- **PhyWorld** — Kang & Yue et al. 2024 · arXiv:2411.02385
- **Paper ckpts** — huggingface.co/quentinll/lewm-pusht & lewm-cube

<br>

**本实验产出**：

- `wm/reports/FINAL_REPORT.md` — 完整报告
- `wm/reports/UNIFORM_MOTION_REPORT.md` + `COLLISION_REPORT.md` — 细节子报告
- `wm/phyworld/scripts/{convert,probe}_*.py` — 数据 + probe 脚本
- `wm/le-wm/train.py` — 加了 `init_from_ckpt` config 支持 paper-init

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Q & A

<br>

**感谢聆听，欢迎提问**

<br>
<br>

完整报告：`wm/reports/FINAL_REPORT.md`
