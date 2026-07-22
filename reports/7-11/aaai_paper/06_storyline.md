# 论文故事线 —— Physics Is Already There: Rethinking Physical Inductive Biases in Latent World Models

**对应稿件**:[paper/main.pdf](paper/main.pdf)(AAAI-27)
**日期**:2026-07-16(**五章重构版**:动机单线化、指标降级进 Setup、三/四章一一对应、留 DINO-WM 数据位)
**用途**:**写论文的唯一蓝本**——章节、论点、数字、图表全按本文档落稿;每个数字的源见 [01_results_ledger.md](01_results_ledger.md) 与 [raw_data/](../../../raw_data/README.md)。

---

## 一句话主线(= 论文 thesis;标题 *Physics Is Already There* 说的就是①)

> **① 物理已经在里面了(= 标题;为什么注入没用)**:latent **已经**把物理**状态**编强了——位置 probe ρ **0.80–0.96**(三域 both-OOD),且**冗余分布在黑盒 190 维**(probe-190 三域实证)。再往共享 latent 注入**同一份状态**只是塞了个**副本**:**不带新信号 → 不提升**;**还占表示容量、分梯度、与预测目标打架 → 有害**。⚠️ 关键(2026-07 干预实验翻案):预测**并没有绕过** slot——它**确实读**(steering / counterfactual patching / Jacobian 三法一致),**但读的是副本**,而读副本给不了表示本就缺的东西(***load-bearing but redundant***)。
>
> **② 那什么才有用(论文的建设性落点,非动机)**:真正的短板**不在"状态",而在预测器的长程 rollout 动力学**(状态可解码、但一自回归就漂)。修它靠**训练协议**——free-rollout 让预测器在训练时暴露于**自身累积误差**、学会纠偏(修 exposure bias)。它**不灌任何物理先验、不碰 latent 结构**,却比一切物理结构注入都管用。

**⚠️ ①② 不是并列的两半**(旧标题 "Why Doesn't Help, and What Does" 曾把它们抬成对偶,已废):① 是论文主体与标题,② 是**从 ① 的机制推出的落点 + 阳性对照**——写作时 ② 服务于 ①,别再写成两条并行主线。

**核心概念**:***load-bearing but redundant***(承载了,但承载的是副本)。两个易混属性必须分开——**load-bearing(承重)**:扰动它会改变预测,**因果**性质,只能靠干预测;**marginal(边际有用)**:它提供了 latent 其余部分没有的信息,**信息**性质,看"加了它预测是否变好"。**probe 两者都证明不了**;而当表示**冗余**时两者恰好分离——正是我们这一例:注入的 slot **承重**(被读)却**不 marginal**(是副本)。
> ⚠️ **旧表述 *decodable but not load-bearing*(presence ≠ use)已作废**:那版基于 probe-190 的**相关性**推断"预测绕过 slot",被 2026-07 的**干预**实验推翻(论文 conclusion 原话:*against our first reading*)。别再用"绕过/旁路"讲机制。

**为什么这版表述值钱**:① 它把"注入物理没用"和"修训练协议有用"**统一进同一个机制**(补 latent **已有的** vs 补 latent **真缺的**),不是两个孤立结论;② 它**免疫"换更高维 slot / 更好映射会不会好"的质疑**——那还是塞**同一份副本**、冗余没消除;要突破须**改架构把这份冗余去掉(extrinsic:物理态是预测唯一必经通道)**,不是换编码。详见 [detail/why_physics_structure_fails.md 层3](detail/why_physics_structure_fails.md)。

**三个不能混的精度点**(写作时必守,否则被审稿人捅穿):
1. **free-rollout 不是"注入物理演化规则",它啥物理都不灌**——手段是改采样协议消 exposure bias;它在论文里的身份是**受控析因变量 + 阳性对照**(§3.3),**不是动机、不是方法贡献**(= scheduled sampling,主动引用)。
2. 物理注入**不是"作用不大",是系统性有害**(30 格 25 格变差、29/30 不优于 baseline、从头共训 Δ+0.558)——反直觉点就在"有害",别软化。
3. 我们**否定的是注入"latent 已有的状态"**,不是所有物理注入;注入 latent **真缺的东西**(动力学/守恒量)或走 **extrinsic 架构**是**未证实的**开放出路(唯一疑似正例 parabola 速度 −0.026,不倚重,future work)。

---

## 全文五章骨架(写论文照此,§3 与 §4 一一对应)

| 章 | 写什么 | 蓝本在下文 |
|---|---|---|
| **1 Introduction** | **动机单线**:物理注入对 latent WM 是否有效?(不掺 free-rollout) | §C1 |
| **2 Related Work** | 物理注入 WM / latent WM / exposure bias 三条线 | §C2 |
| **3 Method** | 我们尝试的各种注入方案 + **每个方案的设计目的** + 协议对照的设计 | §C3(3.1–3.4) |
| **4 Experiments** | Setup(含指标,**一段带过**)+ 结果与分析,**逐小节对应 §3** | §C4(4.0–4.4) |
| **5 Conclusion** | 结论 + limitation + future work | §C5 |

---

## C1 第一章 Introduction(动机,单线)

**只沿一条线走:物理注入在 latent world model 上做得少、没被系统检验——我们来回答它是否有效。** ⚠️ free-rollout 不出现在动机里(它是 §3.3 的析因工具)。

1. **背景现象(hook)**:JEPA 系 latent 世界模型短程预测近乎完美(单步 latent cos 0.98–0.99)、物理状态**可从 latent 线性解码**(位置 probe ρ,both-OOD 判读分区:uniform **0.86–0.96**、parabola 0.83–0.89、collision 0.80–0.89)——**但一自回归 rollout 就偏离动力学**(collision h28 cos 掉到 **0.24**,OOD 更崩),而位置信息仍留在真实表示里(collision both-OOD REAL-emb ρ≈0.84)。→ **状态可解码 ≠ 物理可遵守**。(全 4 分区全距 0.48–0.98,低端几格是 range restriction 统计假象,`check_pos_variance` 实测坐实 → [detail/why_physics_structure_fails.md 层0](detail/why_physics_structure_fails.md);出处 `raw_data/runs/aaai_p0/rollout_{域}_baseline_fr_s1234.log`)

   ![](figures/fig1_thesis_presence_not_use.png)

   > **Fig 1 读图**(collision 域,论文 teaser):横轴 = rollout 步数,纵轴 = 与真实的吻合度(0–1)。**绿虚线** = 从真实帧 latent 线性解出位置的 ρ≈0.84(位置一直在 latent 里、平线);**红线**(teacher-forced 原始训练)单步 0.99 → h28 **崩到 0.24**;**蓝线**(free-rollout,§3.3 的协议对照)长程稳住 0.48。→ **状态可解码 ≠ 物理可遵守(decodable ≠ compliant)**。⚠️ 别把这张图读成"预测不用位置"——干预已证明它**用**(§4.2 ④);图说的是**长程 rollout 仍会漂**,短板在预测器的演化能力,不在状态是否在场。(写 Intro 时蓝线只作预览、不展开)

2. **现状**:面对"rollout 违反物理",一条被寄予厚望的药方是**把物理注入模型**——它在**别的形态**上确有成功:PIWM 系用**专用 extrinsic 架构**(低维物理态是预测必经通道),deep-sup(2504.03861)在**低维状态 latent**(8-D、probe 占 38%)上有效。**但在共享高维 latent 的世界模型上,物理注入只有零散、各测各的尝试,从没被系统检验过。**

3. **核心问题(全文唯一动机)**:**把物理注入 latent world model,能否让它物理一致?** 我们把注入的设计空间扫满(5 机制家族、含变体 10 臂、3 物理域 + 2 照片级仿真基准,~200 训练 run)来回答。

4. **答案预览 + 贡献**:①(负,主体)注入**系统性有害**——30 格 29/30 不优于 baseline,连正确物理形式(a=g)和从头共训都救不回;②(机制)因为 latent **已经**冗余编码了物理状态,注入 = 塞副本 + 抢梯度;干预证明预测**确实读**这个 slot,但读的是副本、给不了新信息(**load-bearing but redundant**),并给出可证伪验证;③(对照落点)同一代码同一指标下,**训练协议**(free-rollout/horizon)带来 2.2–8.3× 提升 → 短板在预测器长程动力学、不在状态;④ extrinsic 架构(PIWM 移植)解决承重但不解决编码器-OOD,必要非充分。

---

## C2 第二章 Related Work(半页~2/3 页,三条线)

- **物理注入的世界模型**:PIWM 2412.12870/2503.02143(extrinsic 专用架构,如实承认其成功、§4.4 移植检验)、deep-sup 2504.03861(**低维状态 latent 上结论正确、用的是可信指标 pred_loss**;我们复核其 recipe 在高维视觉 latent 的适用边界,§4.1)。
- **latent 世界模型与物理评测**:JEPA/LeWM 系;PhyWorld(已证 video-gen 堆数据不行)、Physion/Physion++(**照片级仿真**,勿称 real-world)。
- **rollout 训练与 exposure bias**:Bengio 2015(Scheduled Sampling)、Ranzato 2016(exposure bias)——**主动承认 free-rollout 不新**,我们的用法是把它当受控析因变量(§3.3)。

---

## C3 第三章 Method:注入方案设计(每个方案:是什么 + 为什么这么设计)

### 3.1 物理注入设计空间:5 家族 → 10 臂(30 格的"方案侧")

沿三条正交轴铺满"往共享 latent 注入物理"的设计空间——**硬/软 × 钉状态/钉演化 × 有/无标签**,每族的设计目的:

| 家族 | 臂 | 一句话原理 | **设计目的(为什么设这个方案)** |
|---|---|---|---|
| ① 固定 slot | structpos | 硬钉前 2 维 = 真实位置(`emb[:,:2]≈proprio`) | 最直接的"状态注入":让指定维度承载物理量,检验**硬约束**是否有效 |
| | +pw30(LBR) | structpos + pos_weight=30 承重加权 | **机制检验**:若失效因 slot 占比低(2/192),加权抬占比应能救——可证伪 |
| | +velocity(posvel) | 位置**和速度**都进 slot(×pw30) | 检验"注入的量是否可外推"是否关键(速度在抛体线性可外推) |
| ② 深监督 probe | probe | 线性头从 latent 读出位置(`probe_head(emb)≈proprio`) | 复现 deep-sup(文献最强竞品)recipe,检验**软约束**在高维视觉 latent 是否成立 |
| | +structpos | 软 probe + 硬 structpos 组合 | 检验软硬互补假设 |
| ③ 运动学头 dyn | free MLP | 位置 slot 挂显式方程 `z+v+a`,a=自由小网络 | 从"钉状态"升级到"钉**演化**":给 slot 装动力学 |
| | strict a=g | 同上但 a=可学重力常数 | **严格 PIWM 形式**——堵"你物理形式不对"的质疑 |
| ④ consistency | consistency | 只要求 rollout 位置的差分速度=真值速度 | 不假设 a 的形式,为 collision 冲量(不连续)设计 |
| ⑤ label-free | label-free | 不钉真值,只要求 slot 按二阶动力学平滑演化 | **无标签**自组织——纯视频没有 proprio 时唯一可行的注入 |
| | grounded | label-free 同结构 + 钉真值 | label-free 的有标签对照(隔离"标签有无"变量) |

**轴闭合**:硬(①③)↔软(②④);状态(①②)↔演化(③④⑤);有标签(9 臂)↔无标签(⑤)。→ 任何"换个注入方式会不会好"都落在已测的轴上。另设**从头共训 vs 后训练嫁接**两种训练方案(2×2),堵"要在预训练注入才行"。(全表 → [detail/physics_injection_full_scan.md](detail/physics_injection_full_scan.md))

### 3.2 机制假设与可证伪检验的设计

**假设(the redundancy problem,冗余问题)**:物理 slot 只占 2/192 维;黑盒 190 维**冗余**编码了同一份位置 → 注入的 slot 是 latent **已有信息的副本**。预测**很可能确实会读它**(Test 4 证明确实读),但**读一份副本无法提供表示所缺的信息** → 不带新信号,却照样消耗容量与梯度份额。为它设计**四个**检验,每个都规定了能证伪它的具体结果(↔ 4.2 四条结果):

| # | 检验 | 怎么设计的 | **想证伪什么** |
|---|---|---|---|
| ① | **probe-190**(冗余直接实证) | 把 latent 拆成"slot 2 维 / 黑盒 190 维",各自训 probe 解位置;另设**随机 2 维**作对照 | 若黑盒 190 维单独就解得出位置 → **副本确实存在(冗余成立)**;若解不出 → 冗余主张被证伪。⚠️ 本检验**说不了任一副本是否被"使用"**——那是 ④ 的活 |
| ② | **约束是否真在主导 encoder**(两个独立测量) | (a) 比值 `(λ·probe_loss)/pred_loss` —— 用**加权 loss 值作梯度大小的代理**(非实测梯度范数);(b) encoder 输出的 **Participation Ratio**(有效维度) | 若比值 ≈1 且 PR 不变 → "约束没接进去/太弱"(即 bug 假说);若比值 ≫1 且 PR 塌 → **约束真生效、且在和预测抢容量** |
| ③ | **pos_weight 全曲线**(LBR,剂量-反应) | pos_weight 1→300 扫满 | 若危害不随权重响应 → 梯度份额机制错;若响应但**加权到头仍救不回** → "物理损失不够"不是根因。**扫描改不掉的恰恰是它全程不变的那样东西——那份副本**(这条预注册了 4.4:唯有从构造上去掉副本的 extrinsic 才可能有效) |
| ④ | **干预:slot 承重吗**(2026-07 新增,**翻案的一条**) | ①–③ 都**不含真正要紧的变量:预测到底读哪个通道**。故对训练好的模型做三种干预:**steering**(偏移一个通道解出的位置,到**另一个**通道上读变化,对照 norm 匹配的随机方向)、**counterfactual patching**(把通道换成 donor 轨迹的值,看 rollout 跟随 donor 多远,对照"没有 slot 可换"的模型)、**逐维 Jacobian**(预测位置对每个 latent 坐标求导,**无需设计干预**) | 若 slot **不承载负荷**,三法在它上面都读**零**。实测三法**均非零且一致** → **slot 确实被读**,旧的"绕过"读法被推翻 |

**为什么要设计 ②**:它是回应"**你实现有 bug**"这个致命质疑的**唯一硬手段**——不辩"我检查过代码",而用模型的可观测行为证明约束真生效了、只是方向有害。

### 3.3 训练协议对照:free-rollout(+ horizon 匹配)

**设计目的(两重身份,都不是"新方法")**:
- **受控析因变量**:物理结构从没和"训练协议"这个最大混淆变量做过受控对比——大家默认协议是背景、结构是变量。我们在**同一 baseline** 上把两者摆到一起:把协议先做对(free-rollout),再问结构还有没有增量价值。
- **阳性对照(positive control)**:负结果最怕"你实现有 bug/指标失效"——free-rollout 证明**同一套代码、同一批指标下,别的干预能大幅提升**,把矛头钉死在结构本身。

**分析**:teacher-forcing 训练每步喂真值、模型从没见过自己的误差,部署时却要把带误差的预测喂回去 → 误差累积、长程崩(**exposure bias**,经典问题)。free-rollout(`num_preds=8` 自回归)让模型在训练时就暴露于自身累积误差、学会纠偏——**不灌任何物理**。辅以 **rollout horizon 匹配动力学复杂度**(碰撞/真实数据吃长 rollout,光滑域不吃)。

### 3.4 外部对照与跨 backbone 泛化的设计

- **PIWM 忠实移植**(extrinsic 对照):回应"没有外部方法 baseline"——官方 3-stage(VAE 128-D → 提取器 → 已知方程动力学)搬到 phyworld,检验"extrinsic 架构是否就是答案"。
- **DINO-WM 复制**(跨 backbone 对照,✅ 2026-07-16 落地):同一协议(TF vs FR + 注入臂)搬到冻结 DINOv2 的 JEPA 实例,结论**不依赖 LeWM 单一实现**——FR≫TF 复现(合成 1.39–1.69×+真实 3.99×)、注入 27/30 不优于 baseline。

---

## C4 第四章 Experiments(逐小节对应 §3;图表全在此)

### 4.0 Setup(~0.75 页,指标只一段带过)

- **模型**:LeWM(ViT-tiny,192-D latent,SIGReg);init = PushT backbone → 域内 ID-1k finetune(附 scratch 对照);种子 3072/1234/42(标明哪些 3 种子)。
- **数据**:PhyWorld 三域(uniform/parabola/collision)+ OOD 四分区(ID / r/m-OOD / v-OOD / both-OOD);Physion++(照片级仿真,直训 h64);Physion(zero-shot OCP,仅作旁证)。
- **指标(一段带过,不展开——非主线)**:判决用 **rollout nMSE + pixel PSNR**(外部指标、直接量"预测对不对";两者所有案例同向);**cos/probe-ρ 只作诊断**——它们是训练目标的对偶量,加对应 loss 必涨,当主指标会高估物理结构(实测反转:cos 升 1.50× 而 nMSE 崩到 0.21;我们早期 sweep 亦被带偏后翻案,Fig 5 置附录)。**判读规则**:uniform/collision 取 both-OOD;**parabola 取 r/m-OOD**(both-OOD 有 h28 nMSE 除零爆点)。四把尺子对照与逐案例 → [detail/evaluation_traps.md](detail/evaluation_traps.md)。

### 4.1 注入全扫结果(↔ 3.1):29/30 不优于 baseline

**总判决**:30 格(10 臂 × 3 域)——**25 格明确变差、4 格持平、仅 1 格真小赢 → 29/30 不优于 baseline**;整列 collision(1.33–1.66×)最狠。

**逐家族回答 3.1 的设计目的**(每族:设计想验证什么 → 数据怎么回答):

| 家族(↔3.1) | 设计目的 | **实测结果** | 判决 |
|---|---|---|---|
| ① **固定 slot**(structpos) | 硬约束能否让指定维度承载物理量并改善预测? | slot **确实**承载了(可解码 ρ 0.31→0.96),但预测更差(uniform 1.40×/collision 1.66×) | ❌ **约束生效了,方向有害** |
| ① **+pw30**(加权) | 若失效因占比低(2/192),加权抬占比应能救 | 只把 uniform·both 拉回**持平**,r/m 与 collision 全程救不回、越加越差 | ❌ 机制方向对,**修不了根本**(→4.2③) |
| ① **+posvel**(可外推量) | "注入的量能否外推"是否是关键? | **parabola 唯一真降**(0.122→0.096,三种子零重叠);同一编码 uniform **+0.076**、collision **+0.228** 反而更差 | ⚠️ **1/30 例外 = 机制签名**:需"既承重、又匹配该域动力学",换域即翻号 → **不是可用先验** |
| ② **深监督 probe** | 复现 deep-sup(文献最强竞品)recipe,软约束在高维视觉 latent 是否成立? | 三域全差(uniform 1.28×/collision 1.65×);parabola 单种子 0.115 看似赢,**三种子 0.137±0.027 vs baseline 0.122±0.005 → 持平且均值略高** | ❌ **recipe 不迁移**:deep-sup 的低维状态 latent(probe 占 38%)成立,192-D 视觉 latent(占 2%)失效 |
| ② **probe+structpos** | 软硬互补假设 | 组合(0.125→三种子 0.141)**打不过 structpos 单用**,collision 1.54× | ❌ 互补假设不成立 |
| ③ **运动学头 dyn** | 从"钉状态"升级到"钉**演化**",给 slot 装动力学 | free MLP 三域全差;**严格 a=g(正确物理形式)也伤**(uniform 1.57×) | ❌ **堵死"你物理形式不对"**——形式对了照样伤 |
| ④ **consistency** | 不假设 a 的形式,为 collision 冲量(不连续)专门设计 | 恰恰在目标域 collision **最差之一**(1.63×) | ❌ 为难域定制的方案在该域失败最狠 |
| ⑤ **label-free** | 无标签自组织(纯视频唯一可行的注入) | 三域全差(collision 1.66×) | ❌ |
| ⑤ **grounded**(有标签对照) | 隔离"标签有无"这一变量 | 同结构加了完美标签仍伤,只是 least-bad(collision 1.33×) | ❌ **失效与标签无关 → 是架构性的** |
| **训练方案 2×2** | 堵"要在预训练注入才行" | 从头共训比后训练嫁接**伤得更狠**(uniform Δ +0.035 → **+0.558**) | ❌ 辩护堵死 |

→ **3.1 的每一条设计动机都被数据否定**,且否定方式互相独立(硬/软、状态/演化、有/无标签、形式对/不对、后训练/从头)——**不是某一种实现没做好,是整个设计空间的问题**。

**两条独立于 phyworld·nMSE 的旁证**(换数据集/指标/任务仍成立):
- **迁移(换指标 AUC + 换任务)**:phyworld→Physion zero-shot 上 **pos_weight 0.551 全配置最差**、低于 free-rollout 0.603、连 random 架构先验 0.607 都够不着 → [detail/real_data_physion.md](detail/real_data_physion.md)。
- **照片级仿真(换数据集)**:Physion++ 上 structpos/cons/consacc 的逐场景 rollout nMSE 全部差于纯 FR(如 mass_dominoes 0.058 → 0.45/0.59/0.60,**3–10×**;单种子但差距远超种子噪声带;口径 = per-scenario 聚合,源 `raw_data/runs/physionpp/eval_pp_{struct,cons,consacc}_e20.log`)。⚠️ **probe 族在 Physion++ 上未跑**——已请另一 session 补(P0,见 [NOTE_to_physion_session_gaps.md](NOTE_to_physion_session_gaps.md));补齐前,"照片级仿真同样成立"的措辞只覆盖 slot 与 consistency 两族。

   ![](figures/fig16_physics_injection_scan.png)

   > **Fig 16 读图**(主表):10 行 = 10 个注入臂(按 5 家族分组:`[slot]`固定编码/`[probe]`深监督/`[dyn]`运动学头/`[cons]`一致性/`[free]`无标签);3 列 = 三域判决分区。每格 = **nMSE/baseline 倍数**(括号内原始 nMSE);**颜色 = 判决:红更差、白持平、绿更好**;`†` = 三种子均值(其余单种子),`✓` = 唯一真提升。→ **几乎全红**;30 格 **25 差 / 4 平 / 仅 posvel·parabola 0.76× 真降 = 29/30 不优于 baseline**。

### 4.2 机制结果(↔ 3.2):冗余实证 + 干预翻案(承重但冗余)

**四个检验逐个回答 3.2(编号一一对应)**:

- **① probe-190 → 冗余成立**:黑盒 190 维**单独**解位置 ρ **0.78–0.92**,与全 192 维几乎等高,且**钉了 slot 也不削弱**;随机 2 维对照仅 0.2–0.5 解不出 → **位置是一份冗余、分布式的副本**。(冗余主张未被证伪)⚠️ 但本条**说不了这两份副本哪个被"用"**——见 ④。
- **② 约束真在主导 encoder、且方向有害 → "bug 假说"被排除**:(a) 加权 loss 比 `(λ·probe_loss)/pred_loss` = **15–125×** ≫1(梯度大小的代理,非实测梯度范数);(b) encoder 有效维度(PR)塌 **39–90%**(uniform 41→4);另有 slot 可解码 ρ **0.31→0.96**。→ 不是"梯度没流",而是**流了、还把 encoder 往低维位置子空间猛拽、和预测拔河**。
- **③ pos_weight 全曲线 → 机制方向对,但修不了根本**:危害确实随权重系统响应(剂量-反应成立),但 4 个域×分区**只有 2 个救回持平、从不净增益**;collision 任何权重都救不回、**越加越差** → "物理损失不够"不是根因。**扫描全程改不掉的,正是它自始至终没动过的那样东西——那份副本**。→ 预注册了 4.4:**唯有从构造上去掉副本的 extrinsic 才可能有效**。

- **④ 干预 → slot 确实被读,旧"绕过"读法被推翻(2026-07)**:三法一致——**steering**:偏移 slot 解出的位置,模型自己的黑盒状态跟着动 **1.5–2.0×**(norm 匹配随机方向 ≈0;加性误差 <3%,扰动保持线性);**counterfactual patching**:把 slot 换成 donor 轨迹的值,预测跟随 donor **47–72%**(无 slot 可换的模型仅 1.5–11%);**Jacobian**:注入的两维敏感度排名从 **~99/192 升到第 1–2 名**。三法失效模式互不相交、三种子 × 两个 backbone 一致。
  → **所以预测没有绕过 slot,它读了。**同一批干预还显示**黑盒保留着自己那份**:钉 slot 不削弱黑盒可解码性,且黑盒凭维数占**总 Jacobian 敏感度的 96–99%**。**两个通道都承重,而预测误差纹丝不动**——这正是**冗余假说**的预言,也正是"slot 被饿死"假说给不出的。结论词:***load-bearing but redundant***(承载了,承载的是副本)。

   ![](figures/fig15_bypass_probe190.png)

   > **Fig 15 读图**(冗余直接实证):三个域,每域两根柱——**黑柱 = 全部 192 维解位置**、**蓝柱 = 去掉物理 slot、只用黑盒 190 维解位置**(probe ρ↑)。两柱几乎等高(0.78–0.92)= **黑盒单独就把位置编了进去**;底部红带 = 随机 2 维对照(0.2–0.5)。

   ![](figures/fig8_lbr_ablation.png)

   > **Fig 8 读图**(LBR 剂量-反应):左 = uniform 上 pos_weight 1→300(蓝 both-OOD / 橙 r/m-OOD,虚线带 = baseline);右 = 三域 nMSE/baseline 比值随 pos_weight。→ 加权只把 **uniform·both 拉回持平**(r/m 救不回)、**collision 越加越差**。→ 承重是**机制验证**、**不是修复方法**。

### 4.3 协议对照结果(↔ 3.3):同一代码下 2.2–8.3×

free-rollout 单开关:合成三域 **2.2–3.6×**、真实 Physion++ **8.3×**(均三种子、区间零重叠),**全 4 分区含 ID 都提升**(2.0–4.6×,不是 OOD 补丁);horizon 匹配:np8→28 把 Physion++ h64 nMSE 单调打到 **1/19**(0.280→0.014)、未见拐点。→ **阳性对照成立**(排除实现/指标失效),且**支配变量在训练协议**——只动"怎么训"、不动"latent 里放什么",却碾压 30 格注入。(论据 [detail/free_rollout_evidence.md](detail/free_rollout_evidence.md))

   ![](figures/fig2_free_rollout.png)

   > **Fig 2 读图**:每域两根柱 = teacher-forced vs free-rollout 的 rollout 误差(nMSE↓),柱顶 = 下降倍数:
   >
   > | 域 | TF | FR | 倍数 |
   > |---|---|---|---|
   > | uniform | 0.300 | 0.136 | 2.2× |
   > | parabola(r/m) | 0.443 | 0.122 | 3.6× |
   > | collision | 1.153 | 0.479 | 2.4× |
   > | **Physion++(仿真,h64)** | 1.174 | 0.141 | **8.3×** |

   ![](figures/fig7_realdata_num_preds.png)

   > **Fig 7 读图**(horizon 匹配):Physion++ 直训,不同 num_preds 的 **by-horizon nMSE(log 轴,↓)**。np8 → np20 → np20+scale → **np28+scale** 长程单调下降,**h64 0.280 → 0.014(1/19)、未见拐点**。→ 真实动力学比合成域更吃长 rollout。

### 4.4 外部对照与跨 backbone 泛化(↔ 3.4)

**PIWM(extrinsic)**:移植后学到正确物理、ID/v-OOD 比 LeWM 还准,**但 size/mass-OOD 崩**(ρ 0.33 vs 0.89)——**PIWM 的红利来自架构而非方程,而其 VAE 编码器同样扛不住 OOD** → extrinsic 消除了冗余副本,却不解决"编码器-OOD",**必要非充分**。这同时解释了别人的正结果为何与我们的负结果不矛盾。

   ![](figures/fig9_piwm_vs_lewm.png)

   > **Fig 9 读图**:官方 PIWM(紫,extrinsic)vs LeWM free-rollout(蓝)的 rolled-out 位置 ρ(↑),4 个 OOD 分区:
   >
   > | 分区 | PIWM | LeWM |
   > |---|---|---|
   > | ID | **0.96** | 0.93 |
   > | **r/m-OOD** | 0.33 ⚠️ | **0.89** |
   > | v-OOD | **0.97** | 0.87 |
   > | **both-OOD** | 0.48 ⚠️ | **0.87** |

**✅ DINO-WM 跨 backbone 复核(2026-07-16 落地,~130 run,3 种子)**:同一协议搬到第二个 JEPA 实例——**冻结 `facebook/dinov2-small`(通用 SSL,从没见过 phyworld)+ 可训练 projector adapter + 完全相同的 predictor/losses/eval**。三格全部落地且**同向**:

| 格 | 结果 | 判决 |
|---|---|---|
| ① **TF vs FR**(全 3 合成域 + 真实 physion,3 种子) | uniform 1.39× / parabola 1.69× / collision 1.68× / **Physion++ 3.99×**,区间零重叠 | ✅ FR≫TF **跨模型复现** |
| ② **注入 30 格**(10 臂×3 域) | **27/30 ≥ baseline**(10 差/17 平/3 好);dyn a=g、grounded 在 collision **1.27×** 最害;3 个"好"格全在 uniform | ✅ **注入不是通用杠杆**,与 Fig16 的 29/30 一致 |
| ③ **REAL-emb ρ**(冻结 encoder presence) | 位置 both-OOD ρ **0.951 > LeWM 0.899**;黑盒 blackbox[2:192] ρ 0.951≈all192 | ✅ presence 更强,**旁路跨模型完好** |

**关键增值**:③ 把步1从"我们的模型编了物理"升级为"**冻结的通用 DINOv2、零物理监督、连梯度都没有,照样把位置线性编到 ρ=0.951**"——presence 是通用视觉表示的固有性质,不是训练产物。堵死"换个 encoder 就不一样"。

**唯一异常已闭环**:uniform pos_weight=300 显著变好(0.67×),但 **shuffle control(钉随机目标)+ weakpin(几乎不钉、只加权)同样 0.77×** → 增益是加权的正则效应、与物理内容无关,且 ID 全退化(容量限制签名)。→ 收进评测陷阱:一个真实的 nMSE 改善,归因被对照证伪。诚实边界:dinowm 注入是"无效(噪声内)"而非 LeWM 的"有害",差异归因于冻结架构限制了旁路——与 extrinsic 结论自洽。

详见 [detail/cross_model_dinowm.md](detail/cross_model_dinowm.md);跨模型热力图 [figures/dinowm_injection_heatmap.png](figures/dinowm_injection_heatmap.png)。→ **4.4 措辞可从"LeWM 上的发现"升级为"两个架构差异显著的 JEPA 实例上的共性"。**

### 4.5 小结(一段,呼应 thesis)

同一 latent、同一代码、同一指标:注入"已有的状态"29/30 无效且有害(4.1),因为注入的是副本(4.2);修"真缺的长程动力学"2.2–8.3×(4.3);去掉副本的 extrinsic 必要非充分(4.4)。→ *decodable but not load-bearing*。

---

## C5 第五章 Conclusion(~0.5 页)

1. **结论**:物理状态在 latent 世界模型里**可解码但不承重**;往共享 latent 注入已有状态系统性有害(29/30);真正的杠杆是训练协议(修长程动力学);extrinsic 架构必要非充分。
2. **建设性出路(future work)**:注入 latent **真缺的东西**(动力学/守恒量而非状态)、extrinsic 承重通道 + 鲁棒编码器。
3. **Limitations(诚实列)**:主判决在 LeWM ViT-tiny,**已由冻结 DINOv2 的第二个 JEPA 实例跨 backbone 复核(FR≫TF + 注入失效同向,2026-07-16)**;Physion++ 物理臂单种子(差距 3–10× 远超噪声带,但如实标注);Physion/Physion++ 是**照片级仿真**、非真实视频;30 格中 26 格单种子(高出 baseline 5–20× 种子 std,4 个贴近格已补三种子);posvel·parabola 例外提示"匹配动力学的注入"可能有效但未系统探索。

---

## 论证互锁(写作自检:为什么审稿人拆不散)

- **4.1 负结果**孤立时会被问"实现有 bug?指标不对?" → **4.3 阳性对照**证明同一套代码/指标下别的干预大赢,堵死实现质疑;**4.0 的判决指标选择**(nMSE+pixel,拒绝 cos/probe)堵死指标质疑。
- **4.2 机制**把负结果变成可证伪的科学发现(probe-190 冗余 + LBR 剂量-反应 + 干预证明 slot 确被读),并预言了 4.4:从构造上**去掉这份副本**(extrinsic)才可能有效。
- **4.4 PIWM** 验证预言的同时划出边界(编码器-OOD 未解决),让结论**建设性收尾**而非负结果堆。

关于 parabola 小提升(−0.026):**不当卖点、不当证据支柱**,诚实说法是"整体有害,parabola 偶尔小赢,可能因速度可外推"。

### 最致命的质疑:"结构没用是不是你实现有 bug"

反驳不靠"检查过代码"(苍白),靠**模型可观测行为证明约束真生效了、只是方向有害**:structured_loss 在降、slot 可解码性大涨(0.31→0.96)、probe-190 实测冗余副本、干预证明 slot 确被读、pos_weight 剂量-反应、加权 loss 比 15–125×、危害随强度单调 + free-rollout 阳性对照。六证据逐条数字 → [detail/why_physics_structure_fails.md](detail/why_physics_structure_fails.md)。

---

## 配图速查(做 PPT / 写论文取图)

矢量 `.pdf` 在 [figures/](figures/),脚本 [figures/storyline_figures.py](figures/storyline_figures.py) 一键重画;每张图的数据表+源 → [detail/figures_gallery.md](detail/figures_gallery.md)。

> **时间紧只讲 3 张**:**Fig 1**(钩子)→ **Fig 16**(30 格全废,主体)→ **Fig 2**(协议才是杠杆)。

| 图 | 讲什么 | 论文位置 | 文件 |
|---|---|---|---|
| **Fig 1** | 可解码 ≠ 预测依赖它(teaser) | §1 Intro | `fig1_thesis_presence_not_use.pdf` |
| **Fig 16** | 注入 30 格全扫,29/30 不优于 baseline | §4.1 主表 | `fig16_physics_injection_scan.pdf` |
| **Fig 15** | 冗余实证:位置副本在黑盒 190 维 | §4.2 | `fig15_bypass_probe190.pdf` |
| **Fig 8** | LBR 剂量-反应(可证伪验证) | §4.2 | `fig8_lbr_ablation.pdf` |
| **Fig 2** | free-rollout 跨域 2.2–8.3×(dinowm 1.39–3.99× 已复现) | §4.3 | `fig2_free_rollout.pdf` |
| **Fig 7** | 长 rollout 单调好、无拐点 | §4.3 | `fig7_realdata_num_preds.pdf` |
| **Fig 9** | PIWM extrinsic 对照:必要非充分 | §4.4 | `fig9_piwm_vs_lewm.pdf` |
| **Fig 5** | cos 陷阱(指标为何这么选) | §4.0 Setup/附录 | `fig5_cos_trap.pdf` |

---

## 为什么这个故事能投 AAAI

不靠"新方法"(free-rollout = Scheduled Sampling,主动引用;它是我们的**析因变量与阳性对照**,不是贡献),靠**一个反直觉、有机制、跨机制×域×基准系统验证的科学发现**:

> *物理归纳偏置在共享 latent 世界模型上系统性失效,因为信息可解码但预测不依赖它;真正的杠杆在训练协议。*

加上 PIWM 的架构性归因(必要非充分)与 DINO-WM 跨 backbone 复核(✅),这是一篇诚实、完整、有解释力的实证论文。创新性评估与审稿预案 → [02_story_and_novelty.md](02_story_and_novelty.md);数字总账 → [01_results_ledger.md](01_results_ledger.md)。
