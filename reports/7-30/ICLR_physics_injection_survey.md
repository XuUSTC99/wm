# 物理注入 / 运动学方程注入调研报告
### —— 从 AAAI 负结果到 ICLR 正方法的设计路线

日期：2026-07-30
目标：ICLR（约两个月窗口），在 AAAI 稿（*Load-Bearing but Redundant*）基础上提出**新的物理注入方法**，提升 OOD 与长程 rollout 精度。
调研范围：物理注入 / 运动学方程 / 积分器结构 / 守恒律 / 对称群，重点 2025–2026。

---

## 0. 执行摘要（结论先行）

**一句话结论**：文献里所有**成功的**物理注入，物理模块都在共享视觉 latent 的**外部或下游**；所有**写进共享 latent 坐标**的做法，只有你的 AAAI 稿系统地测过——并且证否了。这不是巧合，有理论原因；而这个理论原因同时给出了新方法应该长什么样。

**理论支点（新发现，是本次调研最重要的收获）**：
Klindt, LeCun, Balestriero 证明 LeJEPA 达到 **linear identifiability——恢复真实隐变量"至多相差一个旋转"**，且成立条件正是 latent 服从高斯分布（而 SIGReg 就是在强制这一点）。
→ 这就是你 AAAI 稿"load-bearing but redundant"的机制解释：**物理状态确实在 latent 里，但是在一个未知旋转过的基底里**。把 dim[0:2] 钉到位置，等于要求这个旋转是轴对齐的——**这是在和 SIGReg 的各向同性先验对抗**。所以你观察到的 effective dim 崩塌（41→4）不是副作用，是两个目标在打架的必然结果。

**由此推出新方法必须满足的三个约束**（全部来自你自己的实验，这是最强的动机链）：

| # | 约束 | 来自你论文的哪个证据 |
|---|---|---|
| C1 | **不能写坐标**（要 basis-free / 旋转等变） | 冗余测试：190 维黑盒解码 ρ=0.92 与全 latent 无差别；+ identifiability 理论 |
| C2 | **不能独占**（不能让低维物理态成为唯一通路） | PIWM port：ID 略胜（0.96/0.97），radius/mass-OOD 崩到 ρ=0.33/0.48 |
| C3 | **必须作用在演化上，不是内容上** | free rollout 2.2–8.3× vs 全部内容注入 26/27 失败 |
| C4 | **必须打赢 free-rollout 基线，不是 teacher forcing** | 你的基线已经强了 2.2–8.3×，这是硬门槛 |

**主推方案（两个组件，一条主线）**：
- **组件 A｜投影运动学约束（Projected Kinematic Constraint, PKC）**：把运动学方程写在**滚动预测**的 latent **二阶差分**上，通过一个满秩线性读出投影，不钉任何坐标。→ 治长程。
- **组件 B｜运动学对称规范化（Kinematic Symmetry Canonicalization, KSC）**：利用运动学方程的**精确时间/空间重标定协变性**，在推理时把 OOD 参数规范化回 ID 区间。→ 治 OOD。

主线句：**"我们不注入解码出的状态，我们用它来做规范化。"**（We do not inject the decoded state; we use it to canonicalize.）
——这把 AAAI 稿的 "presence" 从一个 null result 变成了新方法的**使能条件**。

---

## 1. 你现有设计空间的洞

你 AAAI 稿的三个轴是：**what（state vs evolution）× hard/soft × labeled/label-free**，9 变体 27 格。

但物理注入实际有**两个你没有当成轴的维度**，而它们恰好是决定成败的维度：

**漏掉的轴 1：注入位点（injection site）**

| 位点 | 说明 | 你覆盖了吗 | 代表工作 |
|---|---|---|---|
| ① 共享 latent 坐标（写入） | 钉 dim、probe 监督 | ✅ 全覆盖，**全失败** | 你的 9 变体、zahorodnii deepsup |
| ② 共享 latent 唯一载体 | 物理态是 transition 的唯一输入 | ✅ 作为 external control | PIWM (mao2024piwm) |
| ③ transition **算子**替换 | 用物理积分器/变分原理替代预测器 | ❌ 明确 out of scope | LaWM、Koopman Dreamer、HNN/LNN |
| ④ latent **演化的可微约束** | 约束轨迹的导数结构，不约束坐标 | ❌ **完全空白** | ← **组件 A 的位置** |
| ⑤ 并联/双流（cross-attn 耦合） | 物理独立一支，双向注意力 | ❌ 未测 | Phantom、PhysVideoGenerator |
| ⑥ 下游读出 + 外部积分器 | 冻结 latent 上挂 prober，物理在下游 | ❌ 未测 | SkyJEPA、NewtonGen |
| ⑦ 对称群 / 推理时规范化 | 物理作为群作用，不作为监督目标 | ❌ **完全空白** | ← **组件 B 的位置** |

**漏掉的轴 2：导数阶数 × 基底**

你的注入按"约束了轨迹的第几阶导数"重新排一下，会看到一个空洞：

| 阶数 | 在钉住的 slot 上（轴对齐基底） | 在全 latent 上（basis-free） |
|---|---|---|
| 0 阶（位置 = 内容） | slot / structpos → **失败** | probe 软监督 → **失败** |
| 1 阶（速度） | consistency → **失败 1.27×** | ❌ 未测 |
| 2 阶（加速度 = 运动学方程本体） | 运动学头 z+vΔt+½aΔt² → **失败** | ❌ **未测 ← 组件 A** |

**关键观察**：你测过的"2 阶"全部是在**钉住的 slot 上**做的——所以它同时承担了坐标钉扎的代价（违反 C1）。**"二阶 + basis-free"这一格从未被测过**，而理论恰好预测它是唯一可能成的那一格。这就是新方法的立足点，而且是从你自己的设计空间里推出来的洞，审稿人无法说"为什么不早做"——因为你的 AAAI 稿正是证明了其它格都不行才逼出这一格。

---

## 2. 文献地图（按注入位点组织）

### 位点 ③：算子级 —— 用物理替换 transition

| 工作 | 做法 | 需标签 | 对你的价值 |
|---|---|---|---|
| **LaWM** (arXiv 2605.08279) | 学 latent 离散 Lagrangian（可学对角质量矩阵 + 势能网），推离散 Euler–Lagrange 条件，**下一步 latent 由 DEL solve 本身产出**（4 步残差修正迭代，常速外推初始化）。H=200 归一化 state RMSE **0.0101 vs 无约束 12.373** | ❌ 无 | **必引**。你 Scope 段划出 operator-level 时只引了 cranmer2020lnn(2020)，LaWM 是 2026 的活证据，且动机段和你诊断一致。不建议全 port（带 decoder + 每步 4 次迭代解） |
| **Koopman Dreamer** (arXiv 2607.19719) | transition 换成谱半径有界 (0.85–0.95) 的 rotation-scaling 块，Koopman 视角 | ❌ 无 | 引。它承认 **contraction–dissipation dilemma**（强收缩压掉任务相关高频）——和你 PIWM 教训同构 |
| **Physically Native World Models** (arXiv 2605.00412) | 结构化相空间 z={(q,p)}，Hamiltonian 分解为动能+势能+交互能，含耗散项 | ❌ 无 | **position paper，零实验**。价值：证明社区在**提议** Hamiltonian WM 但**没人验证过共享视觉 latent 上行不行**——这正是你的战场。引它做 "the community proposes, we test" |
| HNN / LNN (greydanus2019, cranmer2020) | 经典 | — | 已引 |

**判断**：算子级是最"重"的一类，两个月做不完一个可信的 port，且你 AAAI 稿已声明 out of scope。**引，不做。** 但组件 A 是它的"轻量版"——见 §3。

### 位点 ④：演化的可微约束 —— **空白，主战场**

没有工作在共享视觉 latent 上把运动学方程写成**滚动轨迹的高阶差分约束**。最接近的：

| 工作 | 差别 |
|---|---|
| **Temporal Straightening** (arXiv 2603.12231, NYU Agentic Learning) | `L_curv = 1 − cos(z_{t+1}−z_t, z_{t+2}−z_{t+1})`。约束二阶差分**方向**，但**无物理**（纯几何平滑）。且只报 planning success，**没报 rollout nMSE** |
| **Controlling Transient Amplification** (arXiv 2605.08856) | 约束 Jacobian 的交换子 + 正规性。有 OOD 实验（KdV 单孤子→多孤子 5000 步，只有正则化模型不崩）。+20–30% 训练时间，**推理零开销，零标签**。明说与 pushforward/noise injection **正交可叠加** |
| **JAWS** (arXiv 2603.05538) | 空间自适应 Jacobian 正则，MAP + 异方差不确定性 |
| **Fang et al.** (arXiv 2501.00195) | SDE 框架证明 modest latent representation error 起隐式正则；提出 Jacobian 正则降 compounding error |

> **⚠ 一个原创理论观察（值得作为副贡献）**
> 匀加速运动学的真实 transition Jacobian 是**切变矩阵** `[[1, Δt],[0,1]]`——它**非正规**（non-normal），但连续两步的 Jacobian **互相交换**（同一个矩阵）。
> 所以对运动学系统，Transient Amplification 那篇的两个罚项一个对一个错：**交换子罚是对的，正规性罚会主动破坏真实物理**。
> 这个观察可以做成 ICLR 的一个小定理 + 消融（交换子 only / 正规性 only / both），是纯分析成本、零额外 run 的贡献点。

### 位点 ⑤：并联双流

| 工作 | 做法 | 对你的价值 |
|---|---|---|
| **Phantom** (arXiv 2604.08503, UIUC) | 双分支 flow matching：视觉支（冻结 Wan2.2）+ 物理支（V-JEPA2 空间，从零训），两层双向 cross-attn 耦合。**"rather than writing physics into shared visual latents"**。VideoPhy PC +50.4% | **极高**。它用 **"recursive loss-weight scheduling"——把 αz 在 0 和正值间循环，以防物理梯度压过共享架构**。这就是你 Table 3 诊断的那个病（loss ratio 3–125×、effective dim 崩 39–90%）！他们经验性地绕过去，**你解释了为什么必须绕**。而且它**完全没有长程/OOD 评估**——你的基础设施正好补这个洞 |
| **PhysVideoGenerator** (arXiv 2601.03665) | 从 noisy diffusion latent 回归 V-JEPA2 物理特征，physics token 经 cross-attn 注入 DiT 的 temporal attention（**独立条件流**） | 引。同样是"不写共享 latent" |

### 位点 ⑥：下游读出 + 外部积分器

| 工作 | 做法 | 对你的价值 |
|---|---|---|
| **SkyJEPA** (arXiv 2606.23444, Rao/Zhang/Balestriero/LeCun/Loianno) | prober 跑在**冻结 latent** 上（encoder+predictor 都不更新），只输出残差加速度 + SO(3) 保结构可微运动学积分器。T=20 步 multi-step rollout loss + SIGReg（**和你同一个正则**）。位置 RMSE 5.56m→**1.43m (3.9×)**，姿态 40.2°→4.71° (8.5×)。500 域随机化（mass ±50%、inertia ±30%）做 OOD | **极高，必引**。这是"**读出而不写入**"的成功案例，与你的 write-in 失败构成同一枚硬币两面。同 JEPA 血统（Balestriero/LeCun） |
| **NewtonGen** (ICLR 2026, arXiv 2509.21309) | 9 维物理态 Z=[x,y,vx,vy,θ,ω,s,l,a]，**线性二阶 ODE + 残差 MLP**：`a_k z̈ + b_k ż + c_k z + d_k + MLP(Z) = 0`。物理态**完全在生成器外部**，转成 optical flow 去 condition 扩散模型 | **必引**。这是"运动学方程注入"目前最强的形式，而且**比你的运动学头更一般**（你是显式 Euler 固定形式，它是可学系数二阶 ODE + 残差）。它的成功恰恰在于物理在外部 |

### 位点 ⑦：对称群 / 推理时规范化 —— **空白，第二战场**

| 工作 | 做法 | 差别 |
|---|---|---|
| **PACE-FNO** (arXiv 2605.18606) | Lie 代数坐标估计器估输入 frame → 映到参考 frame → 跑标准 FNO → 映回。**OOD 相对误差降至 1/12，在 translation 和 Galilean shift 下** | **最接近，但**：作用在 PDE 场（Burgers/浅水/NS）+ FNO + 周期域，物理态**是给定的输入场**。不是视觉 latent，物理态不需要"读出" |
| **Equivariance with Learned Canonicalization** (Kaba et al., ICML 2023) | 学一个 canonicalization 函数，可插进现有架构 | 方法骨架的标准引文 |
| **Test-Time Canonicalization / FoCal** (arXiv 2507.10375)；**Zero-Shot TTC via OOD Scoring** (arXiv 2606.24178) | 推理时把输入映回训练分布，OOD 打分当能量。translation 主导的 shift 上 OOD 误差降至 1/12 | 只做感知（分类/点云），不做动力学 rollout |
| **TAWM** (arXiv 2506.08441, Nhu/Son/Lin) | 世界模型**显式条件化于 Δt**：`z^{t+Δt} = z_t + d(z_t,a_t,Δt)·τ(Δt)`，τ 是对数变换；训练时 Δt 服从 **log-uniform**。对未见观测率鲁棒（Δt=50ms，20× 默认值，成功率仍 ~90%）。明确声称通过"任意时间间隔的单步预测"缓解 compounding error | **组件 B 的直接零件**。而且注意：**Δt 条件化对运动学头是原生的**（z+vΔt+½aΔt² 本来就含 Δt），黑盒预测器要额外加机制 |

**Galilean 相关**：局部坐标系构造 + global-to-frame 转换可诱导 Galilean 与旋转不变性（见 §7 参考）。你的 velocity-OOD 本质就是 Galilean boost。

### 位点补充：接触/碰撞（对你 collision 域）

- **Allen et al., "Graph network simulators can learn discontinuous, rigid contact dynamics"** (CoRL 2022)：**GNS 能从数据里学会不连续刚体接触，不需要专门结构**。
  → 这对你的负结果是**正面支撑**：你为碰撞冲量专门设计的 consistency loss 反而 1.27× 变差，而普通数据驱动模型学接触是学得会的。**"不连续性是可学的，缺的不是结构"**——这句话值得写进 ICLR 的 motivation。
- ContactNets (arXiv 2009.11193)、Contact-Aware Neural Dynamics (arXiv 2601.12796)：接触作为 LCP + 可微求解器。重，不建议做。

### 理论支点

- **When Does LeJEPA Learn a World Model?** (arXiv 2605.26379, Klindt/LeCun/Balestriero)
  LeJEPA 达到 **linear identifiability（恢复隐变量至多一个旋转 Q）**，成立条件是 latent 高斯 + OU 型转移 + 分量独立；latent 违反高斯性（重尾/Laplace/均匀）时 recovery 退化。度量用 rotation-invariant 线性 probe（R²、alignment loss、whitening error、recovery error ‖h(z)−Qz‖²）。**只做单步动力学与有限步最优控制，不测长程 rollout，也不测 OOD。**
  → 三重价值：(1) 给你 AAAI 的冗余现象一个**理论机制**；(2) `Q` 的存在直接论证**为什么必须 basis-free**；(3) 它自己**没测长程和 OOD**，所以你在长程/OOD 上继续这条线是自然延伸而非重复。

- **Beyond Decodability: Reconstructing LM Representations with an Encoding Probe** (arXiv 2605.00607)
  区分 **decoding probe（读）vs encoding probe（写）**。你的 "presence ≠ use" 论证可以借这套术语加固：你测的是 decodability，而注入干的是 encoding——两者不对称，这正是失败的形式化表述。

---

## 3. 主推方案

### 组件 A｜投影运动学约束（PKC）— 治长程

**动机链**：你的表格里"二阶 + basis-free"这一格从未被测；identifiability 理论说 latent 只在旋转意义下确定，所以约束必须 basis-free；你的 free-rollout 结果说杠杆在演化上。三条线交于同一格。

**形式**：设 `W ∈ R^{2×192}` 为满秩线性读出（联合训练，**不用来钉任何坐标**）。对**自回归滚动**的预测序列 `ẑ` 施加：

```
L_PKC = Σ_k ‖ W(ẑ_{t+k+1} − 2 ẑ_{t+k} + ẑ_{t+k−1}) − a_k · Δt² ‖²
```

其中 `a_k` 是加速度：匀速域 `a=0`，抛物域 `a=g`（一个可学常数，或严格给定），碰撞域 `a` 由学习的 MLP(ẑ,u) 给出（冲量式）。

**为什么这躲开了你的负结果**：

| 你的负结果 | PKC 为什么不撞上 |
|---|---|
| 钉坐标与 SIGReg 各向同性对抗 → effective dim 崩 41→4 | `W` 满秩线性 ⟹ 约束对 latent 的任意旋转不变（旋转 z 就旋转 W）。**不预留任何维度，不指定基底** |
| 冗余：190 维已有同一份位置 | PKC 不复制状态。`W ẑ` 的**二阶差分**只说轨迹怎么弯，不说物体在哪 |
| probe 软监督"机械地"抬高 ρ（Trap 1） | 二阶差分**无法**靠"让状态存在"来满足——位置误差 ε 会放大成二阶差分误差 4ε。它专门罚 probe 误差的**高频时间分量** |
| 内容注入不改善长程 | 约束加在 `ẑ`（模型自己滚出来的）而非 `z`（编码的真值），直接正则化长程行为，与 free rollout 天然复合 |

**与 Verlet 的关系（一个漂亮的统一）**：`p_{t+1} = 2p_t − p_{t−1} + aΔt²` 就是 **leapfrog/Verlet 更新**的显式形式。所以 PKC 等价于说：*被投影的 latent 轨迹必须满足 Verlet 积分条件*。而你现在的运动学头用的是**显式 Euler**（`z + vΔt + ½aΔt²`）——**非辛、会漂能量**。这给你一个免费的、干净的消融轴：**积分器选择（Euler vs Verlet/辛）**，而且有明确物理理由预测 Verlet 更好。这条消融本身就值一个小节。

**实现位置**：[train.py:137](../../le-wm/train.py#L137) `pred_emb` 出来之后（shape `(B,H,D)`），做一次二阶差分 + 一个线性层，约 15 行。

### 组件 B｜运动学对称规范化（KSC）— 治 OOD

**核心洞察**：运动学方程有**精确的重标定协变性**：
- 时间重标定 `t → λt` ⟹ `v → v/λ`, `a → a/λ²`
- 空间重标定 `x → μx` ⟹ `v → μv`, `a → μa`

**于是**：一条 `v=6` 的 velocity-OOD 轨迹，在时间重标定下**恒等于**一条 `v=3` 的 ID 轨迹（以 2Δt 采样）。你的 velocity-OOD 不是一个新物理，**它是 ID 物理的一个群作用像**。

**方法**：
1. **条件化**：predictor 显式接收 `(Δt, 空间尺度 s)`。对运动学头这是**原生的**（Δt 本来就在公式里）；对黑盒支按 TAWM 的 `τ(Δt)` 对数变换注入。
2. **训练**：`Δt` 与 `s` 取 log-uniform 范围（TAWM 做法）。
3. **推理时规范化**：用 probe `W` 从 latent 估计当前速度/尺度 → 算出把它拉回 ID 区间所需的群元 `(λ, μ)` → 在规范 frame 里 rollout → 映回目标 frame。

**为什么只有你能做这一步**：估计群元需要知道当前的速度和尺度。你的 AAAI 稿正好证明了**位置/速度从 latent 线性可解码，ρ=0.92（LeWM）/0.95（冻结 DINOv2），且在 both-OOD 分区上仍然成立**。**这就是使能条件。** PACE-FNO 能做规范化是因为它的物理量是输入场里直接给的；你能做是因为你**测量过**它在 latent 里可读——这是你论文最强的实证资产被第一次真正用起来。

**同时解释了为什么你的 temporal-stride 增广会输**（0.556 vs appearance 0.376）：增广给了模型不同 stride 的样本，但**没给它 stride 是多少的信息**，模型无法把观测到的速度和时间尺度解耦。TAWM 的结论正是"条件化才是关键"。这不是补救叙事，是一个可检验的预测——**做一组"增广但不条件化 / 条件化但不增广 / 两者都有"的消融就能验证**。

### 主线论述（ICLR 的 story）

> AAAI 稿证明：把可解码的物理状态**写回** shared latent 无效——状态已经在那儿了（冗余），而且写入在和各向同性先验对抗。
> ICLR 稿主张：可解码性的正确用法不是把状态**再注入一次**，而是 (A) 用它约束**轨迹的微分结构**，(B) 用它估计**对称群元并做规范化**。
> **物理进入模型的方式应该是约束和群作用，而不是监督目标。**

**杀手级验证**：你已经建好了因果干预工具箱（steering / counterfactual patching / per-dim Jacobian / INLP）。你的 AAAI 稿定义了两个性质——**load-bearing**（干预它会改变预测）和 **marginal**（贡献了 latent 其余部分没有的信息）——并证明所有 9 个变体只满足前者。
**ICLR 稿的成果就是拿出第一个 load-bearing AND marginal 的通道。** 词汇表已经建好，评审能直接看懂两篇的接续关系。这是最强的形式。

---

## 4. 消融矩阵（论文骨架）

| 轴 | 取值 | 目的 |
|---|---|---|
| 注入阶数 | 0 阶（=你的 probe）/ 1 阶 / **2 阶(PKC)** | 证明"阶数"是关键轴，且你已有 0 阶的失败数据可直接复用 |
| 基底 | 轴对齐钉扎（=你的 slot）/ **basis-free (W)** | 直接检验 identifiability 推论；已有 slot 失败数据 |
| 积分器 | 显式 Euler（=你现在的头）/ **Verlet(辛)** | 物理上有明确预测，零额外机制 |
| Jacobian 罚 | 无 / 交换子 / 正规性 / 两者 | 检验"运动学要交换子不要正规性"的理论观察 |
| Δt 处理 | 固定 / 增广不条件化 / **条件化 + log-uniform** | 解释 temporal-stride 增广为何输 |
| 推理规范化 | 关 / **开 (KSC)** | 主 OOD 结果 |
| 位点对照 | in-latent(=AAAI) / 双流(Phantom 式) / 下游读出(SkyJEPA 式) / **PKC+KSC** | 位点是决定性维度的证据 |
| Backbone | LeWM / 冻结 DINOv2 | 跨骨干复现（你已有全套） |
| 域 | uniform / parabola / collision / Physion++ | 已有 |

**基线必须是 free rollout（C4）**，且沿用你的 3 seed（3072/1234/42）+ 四分区 + nMSE/PSNR 双指标 + Trap 检查清单。**你的评估协议本身就是护城河**——审稿人很难攻击一个自带 5 个 metric pathology 目录的评估。

---

## 5. 两个月排期建议

| 周 | 内容 | 产出/止损点 |
|---|---|---|
| W1 | PKC 最小实现（15 行）+ 2 阶 basis-free 单域快跑（uniform, 3 seed） | **止损点 1**：若不打赢 free rollout，立刻转 §6 备选 B1 |
| W2 | PKC 三域 × 3 seed；Verlet vs Euler 消融 | 主表第一块 |
| W3 | Δt 条件化 + log-uniform 训练（TAWM 式），velocity-OOD 上跑 | **止损点 2**：条件化是否让 velocity-OOD 动起来 |
| W4 | KSC 推理时规范化（用现成 probe 估群元）；radius/mass 轴 | 主 OOD 结果 |
| W5 | Physion++ 移植（PKC + KSC），horizon 64 | 真实性证据 |
| W6 | 因果干预：证明新通道 load-bearing **AND marginal**（复用现成工具箱） | 机制章 |
| W7 | 位点对照组：双流 Phantom 式 + 下游 SkyJEPA 式（各一个简化实现） | 位点维度的证据 |
| W8 | 冻结 DINOv2 跨骨干复现 + Jacobian 罚消融 + 写稿 | 完稿 |

**run 预算估计**：PKC 主体 3 域×3 seed×(2 阶数 + 2 积分器) ≈ 36；KSC ≈ 27；位点对照 ≈ 18；跨骨干 ≈ 30；Physion++ ≈ 12。合计约 **120–150 run**，在你 AAAI 的 ~330 run（200 + 130）量级内，两个月可行。

---

## 6. 备选方案（按优先级，止损后切换）

- **B1｜下游读出 + 外部运动学积分器（SkyJEPA 式）**：latent 冻结，prober 读物理态，外部二阶 ODE（NewtonGen 式可学系数 + 残差 MLP）滚动，再融合回预测。**风险最低**——文献里两个独立成功案例，且完美绕开 C1/C2。缺点：新颖性偏低，需要靠 OOD/长程的系统评估和你的干预工具箱来撑。
- **B2｜并联双流 + 你的 loss-ratio 诊断做调度**：Phantom 式双分支，但**用你 Table 3 的 loss ratio 作为自动调度信号**（Phantom 是手工循环 αz）。这是一个"用诊断驱动设计"的干净贡献。
- **B3｜Jacobian 交换子正则 + 运动学理论分析**：纯正则化，无标签，推理零开销，+20–30% 训练时间。配上"运动学要交换子不要正规性"的定理，可以独立成一个小而美的贡献。
- **B4｜LaWM 轻量版**：只把 DEL residual 当 auxiliary loss，不做 4 步 solve。中等成本，能蹭上变分原理的叙事。

---

## 7. 风险清单

1. **PKC 的 `W` 可能退化**：若 `W` 与 predictor 联合训练，可能学到让约束平凡满足的 `W`。→ 对策：`W` 用**冻结的**、事先在 encoder 上拟合好的 ridge probe（你已有这套代码和协议），或加满秩/正交约束。**这一条必须在 W1 就验证**。
2. **组件 B 的群元估计误差会被放大**：`λ` 估错会直接错配时间尺度。→ 对策：先用 ground-truth 群元做 oracle 上界实验，确认天花板值得追，再换成 probe 估计。
3. **content-free control 是必须的**：你 Trap 5 已经吃过一次亏（weight 300 的"胜利"被 shuffle 控制组瓦解）。PKC 必须配 shuffled-target 和 wrong-acceleration 控制组，否则审稿人会直接照搬你自己的 Trap 5 来打你。
4. **C4 门槛**：你的 free-rollout 基线太强了（已经 2.2–8.3×）。要准备好一个可能的结果是"PKC 与 free rollout 复合后再降 1.2–1.5×"——这不算大，但如果在**每个 seed、每个分区、两个骨干上都一致**，配上机制章的 marginal 证明，是可以发的。**提前想好怎么讲"小而稳健且机制清楚"的故事**，别赌一个大数字。
5. **两篇的关系要写清**：AAAI 稿（若中）是 diagnosis，ICLR 是 cure。要在 ICLR 里明确引用并且不能自我抄袭；若 AAAI 未中，ICLR 需要自包含地重述必要的负结果——这实际上更好讲，但篇幅要预留。

---

## 8. 主要参考文献（本次调研已核实）

**理论支点**
- Klindt, LeCun, Balestriero. *When Does LeJEPA Learn a World Model?* arXiv 2605.26379
- *Beyond Decodability: Reconstructing LM Representations with an Encoding Probe.* arXiv 2605.00607

**算子级 / 变分**
- *LaWM: Least Action World Models.* arXiv 2605.08279
- Li et al. *Koopman Dreamer: Spectrally Constrained Latent Dynamics.* arXiv 2607.19719
- Cui, Ma. *Physically Native World Models: A Hamiltonian Perspective.* arXiv 2605.00412（position paper，无实验）

**运动学方程 / 二阶 ODE**
- *NewtonGen: Physics-Consistent and Controllable T2V via Neural Newtonian Dynamics.* ICLR 2026, arXiv 2509.21309
- Rao, Zhang, Balestriero, LeCun, Loianno. *SkyJEPA.* arXiv 2606.23444

**并联双流**
- Shen, Xiong, Yu, Lourentzou. *Phantom: Physics-Infused Video Generation.* arXiv 2604.08503
- *PhysVideoGenerator: Latent Physics Guidance.* arXiv 2601.03665

**演化约束 / Jacobian**
- *Controlling Transient Amplification Improves Long-horizon Rollouts.* arXiv 2605.08856
- *JAWS: Spatially-Adaptive Jacobian Regularization.* arXiv 2603.05538
- Fang, Du, Wang, Zhang. *Towards Unraveling and Improving Generalization in World Models.* arXiv 2501.00195
- *Temporal Straightening for Latent Planning.* arXiv 2603.12231
- *PDE-Refiner.* NeurIPS 2023

**对称群 / 规范化 / Δt**
- *PACE-FNO: Physics-Aligned Canonical Equivariant FNO under Symmetry-Induced Shifts.* arXiv 2605.18606
- Kaba et al. *Equivariance with Learned Canonicalization Functions.* ICML 2023, arXiv 2211.06489
- *Lie Algebra Canonicalization: Equivariant Neural Operators under arbitrary Lie Groups.* ICLR 2025, arXiv 2410.02698
- *Test-Time Canonicalization by Foundation Models (FoCal).* arXiv 2507.10375
- *Zero-Shot Test-Time Canonicalization using OOD Scoring.* arXiv 2606.24178
- Nhu, Son, Lin. *Time-Aware World Model (TAWM).* arXiv 2506.08441

**接触 / 不连续**
- Allen et al. *Graph network simulators can learn discontinuous, rigid contact dynamics.* CoRL 2022
- Pfrommer et al. *ContactNets.* arXiv 2009.11193
- *Contact-Aware Neural Dynamics.* arXiv 2601.12796

**测试时自适应**
- Wang, Bounou, LeCun, Ren. *AdaJEPA: An Adaptive Latent World Model.* arXiv 2606.32026

**增广 vs 等变**
- *To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking.* arXiv 2510.01349
- *Exact equivariance, kept through training, buys zero-shot generalisation.* arXiv 2606.03003
