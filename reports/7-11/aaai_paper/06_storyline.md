# 论文故事线 —— Toward Physics-Consistent Latent World Models: Why Injecting Physics Doesn't Help, and What Does

**对应稿件**:[paper/main.pdf](paper/main.pdf)(AAAI-27)
**日期**:2026-07-16(**五章重构版**:动机单线化、指标降级进 Setup、三/四章一一对应、留 DINO-WM 数据位)
**用途**:**写论文的唯一蓝本**——章节、论点、数字、图表全按本文档落稿;每个数字的源见 [01_results_ledger.md](01_results_ledger.md) 与 [raw_data/](../../../raw_data/README.md)。

---

## 一句话主线(= 论文 thesis,直接对应标题两半)

> **Why injecting physics doesn't help**:latent **已经**把物理**状态**编强了——位置 probe ρ **0.80–0.96**(三域 both-OOD),且**冗余分布在黑盒 190 维**(probe-190 三域实证)。再往共享 latent 注入**同一份状态**只是塞冗余:**不带新信号 → 不提升**;**还占表示容量、分梯度、与预测目标打架 → 有害**;而预测大可绕过物理 slot、直接走黑盒那份(***not load-bearing***)。
>
> **What does help**:真正的短板**不在"状态",而在预测器的长程 rollout 动力学**(状态可解码、但一自回归就漂)。修它靠**训练协议**——free-rollout 让预测器在训练时暴露于**自身累积误差**、学会纠偏(修 exposure bias)。它**不灌任何物理先验、不碰 latent 结构**,却比一切物理结构注入都管用。

**核心概念**:***decodable but not load-bearing***(存在 ≠ 使用,presence ≠ use)——decodable = 信息在场,load-bearing = 预测真的靠它。全文发现都挂在这个落差上。

**为什么这版表述值钱**:① 它把"注入物理没用"和"修训练协议有用"**统一进同一个机制**(补 latent **已有的** vs 补 latent **真缺的**),不是两个孤立结论;② 它**免疫"换更高维 slot / 更好映射会不会好"的质疑**——那还是塞冗余状态、没堵旁路;要突破须**改架构堵旁路(extrinsic)**,不是换编码。详见 [detail/why_physics_structure_fails.md 层3](detail/why_physics_structure_fails.md)。

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

   > **Fig 1 读图**(collision 域,论文 teaser):横轴 = rollout 步数,纵轴 = 与真实的吻合度(0–1)。**绿虚线** = 从真实帧 latent 线性解出位置的 ρ≈0.84(位置一直在 latent 里、平线);**红线**(teacher-forced 原始训练)单步 0.99 → h28 **崩到 0.24**;**蓝线**(free-rollout,§3.3 的协议对照)长程稳住 0.48。→ **信息在场 ≠ 预测用它(presence ≠ use)**。(写 Intro 时蓝线只作预览、不展开)

2. **现状**:面对"rollout 违反物理",一条被寄予厚望的药方是**把物理注入模型**——它在**别的形态**上确有成功:PIWM 系用**专用 extrinsic 架构**(低维物理态是预测必经通道),deep-sup(2504.03861)在**低维状态 latent**(8-D、probe 占 38%)上有效。**但在共享高维 latent 的世界模型上,物理注入只有零散、各测各的尝试,从没被系统检验过。**

3. **核心问题(全文唯一动机)**:**把物理注入 latent world model,能否让它物理一致?** 我们把注入的设计空间扫满(5 机制家族、含变体 10 臂、3 物理域 + 2 照片级仿真基准,~200 训练 run)来回答。

4. **答案预览 + 贡献**:①(负,主体)注入**系统性有害**——30 格 29/30 不优于 baseline,连正确物理形式(a=g)和从头共训都救不回;②(机制)因为 latent **已经**冗余编码了物理状态,注入 = 塞冗余 + 抢梯度,预测走黑盒旁路(decodable but not load-bearing),并给出可证伪验证;③(对照落点)同一代码同一指标下,**训练协议**(free-rollout/horizon)带来 2.2–8.3× 提升 → 短板在预测器长程动力学、不在状态;④ extrinsic 架构(PIWM 移植)解决承重但不解决编码器-OOD,必要非充分。

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

**假设(load-bearing problem)**:物理 slot 只占 2/192 维、~1% 梯度;黑盒 190 维**冗余**编码了同一份位置 → 预测绕过 slot 走黑盒。为它设计两个检验:
- **probe-190(旁路直接实证)**:把 latent 拆成"slot 2 维 / 黑盒 190 维",分别解位置——若黑盒单独就能解出,旁路存在。
- **LBR 全曲线(剂量-反应)**:pos_weight 1→300 扫满——若机制方向对,危害应随权重系统响应;若加权也救不回,证明旁路是架构性的、非"注入量不够"。

### 3.3 训练协议对照:free-rollout(+ horizon 匹配)

**设计目的(两重身份,都不是"新方法")**:
- **受控析因变量**:物理结构从没和"训练协议"这个最大混淆变量做过受控对比——大家默认协议是背景、结构是变量。我们在**同一 baseline** 上把两者摆到一起:把协议先做对(free-rollout),再问结构还有没有增量价值。
- **阳性对照(positive control)**:负结果最怕"你实现有 bug/指标失效"——free-rollout 证明**同一套代码、同一批指标下,别的干预能大幅提升**,把矛头钉死在结构本身。

**分析**:teacher-forcing 训练每步喂真值、模型从没见过自己的误差,部署时却要把带误差的预测喂回去 → 误差累积、长程崩(**exposure bias**,经典问题)。free-rollout(`num_preds=8` 自回归)让模型在训练时就暴露于自身累积误差、学会纠偏——**不灌任何物理**。辅以 **rollout horizon 匹配动力学复杂度**(碰撞/真实数据吃长 rollout,光滑域不吃)。

### 3.4 外部对照与跨 backbone 泛化的设计

- **PIWM 忠实移植**(extrinsic 对照):回应"没有外部方法 baseline"——官方 3-stage(VAE 128-D → 提取器 → 已知方程动力学)搬到 phyworld,检验"extrinsic 架构是否就是答案"。
- **DINO-WM 复制**(跨 backbone 对照,⏳ 另一 session 在跑):同一协议(TF vs FR + 注入臂)搬到另一个 JEPA 系 latent WM,检验结论**不依赖 LeWM 单一实现**。

---

## C4 第四章 Experiments(逐小节对应 §3;图表全在此)

### 4.0 Setup(~0.75 页,指标只一段带过)

- **模型**:LeWM(ViT-tiny,192-D latent,SIGReg);init = PushT backbone → 域内 ID-1k finetune(附 scratch 对照);种子 3072/1234/42(标明哪些 3 种子)。
- **数据**:PhyWorld 三域(uniform/parabola/collision)+ OOD 四分区(ID / r/m-OOD / v-OOD / both-OOD);Physion++(照片级仿真,直训 h64);Physion(zero-shot OCP,仅作旁证)。
- **指标(一段带过,不展开——非主线)**:判决用 **rollout nMSE + pixel PSNR**(外部指标、直接量"预测对不对";两者所有案例同向);**cos/probe-ρ 只作诊断**——它们是训练目标的对偶量,加对应 loss 必涨,当主指标会高估物理结构(实测反转:cos 升 1.50× 而 nMSE 崩到 0.21;我们早期 sweep 亦被带偏后翻案,Fig 5 置附录)。**判读规则**:uniform/collision 取 both-OOD;**parabola 取 r/m-OOD**(both-OOD 有 h28 nMSE 除零爆点)。四把尺子对照与逐案例 → [detail/evaluation_traps.md](detail/evaluation_traps.md)。

### 4.1 注入全扫结果(↔ 3.1):29/30 不优于 baseline

30 格(10 臂 × 3 域)判决:**25 格明确变差、4 格持平、仅 1 格真小赢**;整列 collision(1.33–1.66×)最狠;**严格 a=g 也伤**(uniform 1.57×)、**从头共训伤得更狠**(uniform Δ +0.035 → +0.558)——3.1 里每条设计动机都被数据否定。唯一例外 posvel·parabola(0.122→0.096,三种子零重叠):速度在抛体里线性可外推——**是"结构既承重又匹配该域动力学才有用"的机制签名,不是可用方法**(同一编码 uniform +0.076 / collision +0.228 反而更差)。**独立于 nMSE 的旁证**:phyworld→Physion zero-shot 迁移上 **pos_weight 0.551 全配置最差**、低于 free-rollout 0.603、连 random 架构先验 0.607 都够不着——换数据集、换指标(AUC)、换任务,"物理结构越强越差"依然成立([detail/real_data_physion.md](detail/real_data_physion.md))。**Physion++ 上同样成立**:structpos/cons/consacc 的逐场景 rollout nMSE 全部差于纯 FR(如 mass_dominoes 0.058 → 0.45/0.59/0.60,3–10×;单种子但差距远超种子噪声带;口径 = per-scenario 聚合,源 `raw_data/runs/physionpp/eval_pp_{struct,cons,consacc}_e20.log`)。

   ![](figures/fig16_physics_injection_scan.png)

   > **Fig 16 读图**(主表):10 行 = 10 个注入臂(按 5 家族分组:`[slot]`固定编码/`[probe]`深监督/`[dyn]`运动学头/`[cons]`一致性/`[free]`无标签);3 列 = 三域判决分区。每格 = **nMSE/baseline 倍数**(括号内原始 nMSE);**颜色 = 判决:红更差、白持平、绿更好**;`†` = 三种子均值(其余单种子),`✓` = 唯一真提升。→ **几乎全红**;30 格 **25 差 / 4 平 / 仅 posvel·parabola 0.76× 真降 = 29/30 不优于 baseline**。

### 4.2 机制结果(↔ 3.2):旁路实证 + 可证伪验证通过(但救不了)

- **probe-190**:黑盒 190 维单独解位置 ρ 0.78–0.92,与全 192 维几乎等高;随机 2 维对照 0.2–0.5 解不出 → **位置冗余铺在黑盒里,旁路存在**。
- **梯度打架**:(λ·probe_loss)/pred_loss = **15–125×**(加权 loss 值作梯度大小代理,非实测梯度范数);encoder 有效维度(PR)塌 39–90%(uniform 41→4)。
- **LBR 全曲线**:危害确实随 pos_weight 系统响应(机制方向对),但 4 个域×分区只有 2 个救回**持平**、从不净增益;collision 任何权重都救不回、越加越差 → **旁路是架构性的,加权修不了根本**。

   ![](figures/fig15_bypass_probe190.png)

   > **Fig 15 读图**(旁路直接实证):三个域,每域两根柱——**黑柱 = 全部 192 维解位置**、**蓝柱 = 去掉物理 slot、只用黑盒 190 维解位置**(probe ρ↑)。两柱几乎等高(0.78–0.92)= **黑盒单独就把位置编了进去**;底部红带 = 随机 2 维对照(0.2–0.5)。

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

**PIWM(extrinsic)**:移植后学到正确物理、ID/v-OOD 比 LeWM 还准,**但 size/mass-OOD 崩**(ρ 0.33 vs 0.89)——**PIWM 的红利来自架构而非方程,而其 VAE 编码器同样扛不住 OOD** → extrinsic 解决"承重/旁路",不解决"编码器-OOD",**必要非充分**。这同时解释了别人的正结果为何与我们的负结果不矛盾。

   ![](figures/fig9_piwm_vs_lewm.png)

   > **Fig 9 读图**:官方 PIWM(紫,extrinsic)vs LeWM free-rollout(蓝)的 rolled-out 位置 ρ(↑),4 个 OOD 分区:
   >
   > | 分区 | PIWM | LeWM |
   > |---|---|---|
   > | ID | **0.96** | 0.93 |
   > | **r/m-OOD** | 0.33 ⚠️ | **0.89** |
   > | v-OOD | **0.97** | 0.87 |
   > | **both-OOD** | 0.48 ⚠️ | **0.87** |

**⏳ DINO-WM(跨 backbone,数据位——另一 session 在跑,回来填)**:同一协议搬到 DINO-WM,证明结论不依赖 LeWM 单一实现。**待填三格**:
| 待填 | 预期形态 |
|---|---|
| ① TF vs FR(至少 1 域) | nMSE 对照 + 倍数(对应 Fig 2 加一行/一组柱) |
| ② structpos 一臂 vs FR baseline | nMSE 判决格(对应 Fig 16 补一行注记) |
| ③(可选)probe REAL-emb ρ | presence 是否同样成立 |
> 填入后 4.4 的措辞从"LeWM 上的发现"升级为"JEPA 系 latent WM 的共性";若 DINO-WM 结果**不同向**,如实写成边界条件(哪类 backbone 逃过 load-bearing problem)——也是有价值的发现,勿硬凹。

### 4.5 小结(一段,呼应 thesis)

同一 latent、同一代码、同一指标:注入"已有的状态"29/30 无效且有害(4.1),因为旁路(4.2);修"真缺的长程动力学"2.2–8.3×(4.3);堵旁路的 extrinsic 必要非充分(4.4)。→ *decodable but not load-bearing*。

---

## C5 第五章 Conclusion(~0.5 页)

1. **结论**:物理状态在 latent 世界模型里**可解码但不承重**;往共享 latent 注入已有状态系统性有害(29/30);真正的杠杆是训练协议(修长程动力学);extrinsic 架构必要非充分。
2. **建设性出路(future work)**:注入 latent **真缺的东西**(动力学/守恒量而非状态)、extrinsic 承重通道 + 鲁棒编码器。
3. **Limitations(诚实列)**:主判决在单 backbone(LeWM ViT-tiny;**DINO-WM 验证 ⏳ 填入后此条可弱化**);Physion++ 物理臂单种子(差距 3–10× 远超噪声带,但如实标注);Physion/Physion++ 是**照片级仿真**、非真实视频;30 格中 26 格单种子(高出 baseline 5–20× 种子 std,4 个贴近格已补三种子);posvel·parabola 例外提示"匹配动力学的注入"可能有效但未系统探索。

---

## 论证互锁(写作自检:为什么审稿人拆不散)

- **4.1 负结果**孤立时会被问"实现有 bug?指标不对?" → **4.3 阳性对照**证明同一套代码/指标下别的干预大赢,堵死实现质疑;**4.0 的判决指标选择**(nMSE+pixel,拒绝 cos/probe)堵死指标质疑。
- **4.2 机制**把负结果变成可证伪的科学发现(probe-190 旁路 + LBR 剂量-反应),并预言了 4.4:堵旁路(extrinsic)才可能有效。
- **4.4 PIWM** 验证预言的同时划出边界(编码器-OOD 未解决),让结论**建设性收尾**而非负结果堆。

关于 parabola 小提升(−0.026):**不当卖点、不当证据支柱**,诚实说法是"整体有害,parabola 偶尔小赢,可能因速度可外推"。

### 最致命的质疑:"结构没用是不是你实现有 bug"

反驳不靠"检查过代码"(苍白),靠**模型可观测行为证明约束真生效了、只是方向有害**:structured_loss 在降、slot 可解码性大涨(0.31→0.96)、probe-190 实测旁路、pos_weight 剂量-反应、加权 loss 比 15–125×、危害随强度单调 + free-rollout 阳性对照。六证据逐条数字 → [detail/why_physics_structure_fails.md](detail/why_physics_structure_fails.md)。

---

## 配图速查(做 PPT / 写论文取图)

矢量 `.pdf` 在 [figures/](figures/),脚本 [figures/storyline_figures.py](figures/storyline_figures.py) 一键重画;每张图的数据表+源 → [detail/figures_gallery.md](detail/figures_gallery.md)。

> **时间紧只讲 3 张**:**Fig 1**(钩子)→ **Fig 16**(30 格全废,主体)→ **Fig 2**(协议才是杠杆)。

| 图 | 讲什么 | 论文位置 | 文件 |
|---|---|---|---|
| **Fig 1** | 可解码 ≠ 预测依赖它(teaser) | §1 Intro | `fig1_thesis_presence_not_use.pdf` |
| **Fig 16** | 注入 30 格全扫,29/30 不优于 baseline | §4.1 主表 | `fig16_physics_injection_scan.pdf` |
| **Fig 15** | 旁路实证:位置冗余在黑盒 190 维 | §4.2 | `fig15_bypass_probe190.pdf` |
| **Fig 8** | LBR 剂量-反应(可证伪验证) | §4.2 | `fig8_lbr_ablation.pdf` |
| **Fig 2** | free-rollout 跨域 2.2–8.3×(+DINO-WM 待填行) | §4.3 | `fig2_free_rollout.pdf` |
| **Fig 7** | 长 rollout 单调好、无拐点 | §4.3 | `fig7_realdata_num_preds.pdf` |
| **Fig 9** | PIWM extrinsic 对照:必要非充分 | §4.4 | `fig9_piwm_vs_lewm.pdf` |
| **Fig 5** | cos 陷阱(指标为何这么选) | §4.0 Setup/附录 | `fig5_cos_trap.pdf` |

---

## 为什么这个故事能投 AAAI

不靠"新方法"(free-rollout = Scheduled Sampling,主动引用;它是我们的**析因变量与阳性对照**,不是贡献),靠**一个反直觉、有机制、跨机制×域×基准系统验证的科学发现**:

> *物理归纳偏置在共享 latent 世界模型上系统性失效,因为信息可解码但预测不依赖它;真正的杠杆在训练协议。*

加上 PIWM 的架构性归因(必要非充分)与(⏳)DINO-WM 跨 backbone 复核,这是一篇诚实、完整、有解释力的实证论文。创新性评估与审稿预案 → [02_story_and_novelty.md](02_story_and_novelty.md);数字总账 → [01_results_ledger.md](01_results_ledger.md)。
