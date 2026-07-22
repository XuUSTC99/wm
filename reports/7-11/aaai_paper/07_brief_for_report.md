# Physics Is Already There
### Rethinking Physical Inductive Biases in Latent World Models — 汇报精简版

> **一句话**:物理信息**已经存在**于世界模型的 latent 里(可解码),但预测**不依赖你注入的那一份**——黑盒维度早已冗余编码了同一份位置,预测绕过注入的物理 slot。所以往共享 latent 上嫁接物理**状态**只是塞冗余(不提升、反有害);**真正让模型遵守物理的是训练协议,不是结构先验。**
>
> 核心概念:***decodable but not load-bearing***(可解码 ≠ 预测靠它 / presence ≠ use)。

*完整版见 [06_storyline.md](06_storyline.md);每个数字的出处/detail 见各 [detail/](detail/) 文档与 [01_results_ledger.md](01_results_ledger.md)。*

---

## 0. 一张图看懂问题:状态可解码,但 rollout 不遵守物理

未注入任何物理的 baseline,编码器本身就把位置编进了 latent(真实帧 probe 解码 ρ **0.80–0.96**,三域 both-OOD)、单步预测近乎完美(cos 0.99)——**但一自回归 rollout 就崩**(collision latent cos 到 h28 掉到 **0.24**)。信息在场,预测却滚不住它。

![](figures/fig1_thesis_presence_not_use.png)

> **图说明**(collision 域):横轴 = rollout 步数(horizon),纵轴 = 与真实的吻合度(0–1,越高越准)。
> - **绿虚线**:从真实帧的 latent **线性解出位置**的相关 ρ≈0.84——**位置信息一直在 latent 里、可读出**(不随 rollout 变,故是平线)。这是 *presence*(信息在场)。
> - **两条曲线**:自回归 rollout **预测出的 latent** 与真实 latent 的余弦。**红线**(teacher-forced,LeWM 原始训练)从单步近乎完美(0.99)一路滚到 h28 **崩到 0.24**——预测滚不住它自己编码的物理;**蓝线**(free-rollout,我们的默认)长程稳住(h28 仍 0.48)。
> - **一句话**:绿线高(状态可解码)、红线崩(预测不遵守它)——**信息"在场"≠预测"用它"(presence ≠ use)**;而修复它的不是注入物理,是换训练协议(红→蓝)。

---

## 1. 发现一(负,主体):物理结构不是通用杠杆

把物理量注入共享 latent 的**整个设计空间扫满**——5 家族 × 含变体 10 臂 × 3 域 = **30 格**,沿"硬/软 × 状态/演化 × 有/无标签"三正交轴铺开:

- **25 格明确变差、4 格持平(3 格三种子后回落 = 种子噪声)、仅 1 格真小赢** → **29/30 不优于 baseline**(唯一赢的 posvel·parabola,是"① slot 占比高到不被旁路绕过 + ② 编的速度在抛体里可外推"两条同时撞上的机制签名、非通用方法)。
- 连**正确的物理形式**(严格重力 a=g)、**从头共训**都救不回 → 堵死"要在预训练注入才行"的辩护。从头共训三种子:uniform **+0.435**[+0.276,+0.594]、collision **+0.147**[+0.074,+0.220] 确证有害,parabola −0.029 持平(该域基线跨度 0.343–0.666、太吵)。**持平不是 rescue**——要推翻这条辩护,注入必须在从头共训下变好,而无一格如此。
- **两条独立旁证**:① phyworld→Physion zero-shot 迁移上 **pos_weight 0.551 全配置最差**、低于 free-rollout 0.603、连 random 架构先验 0.607 都够不着——换数据集/指标(AUC)/任务仍是"结构越强越差";② **Physion++(照片级仿真)上同样成立**:structpos/cons/consacc 的逐场景 rollout nMSE 全部 3–10× 差于纯 FR(如 mass_dominoes 0.058 → 0.45/0.59/0.60)。

![](figures/fig16_physics_injection_scan.png)

> **图说明**:纵向 10 行 = 物理注入的 10 个臂(按 5 家族分组,`[slot]`固定编码 / `[probe]`深监督 / `[dyn]`运动学头 / `[cons]`一致性 / `[free]`无标签);横向 3 列 = 三个域的判决分区。每格 = 该臂的 **nMSE / baseline 倍数**(括号内为原始 nMSE);**颜色=判决:红更差、白持平、绿更好**。`†`=该格用三种子均值(其余单种子 3072),`✓`=唯一真提升。→ **几乎全红**:整列 collision(1.33–1.66×)最狠;30 格里 **25 差、4 平、仅 posvel·parabola 0.76× 真降(†✓)= 29/30 不优于 baseline**。

*(全表+逐臂来源 → [detail/physics_injection_full_scan.md](detail/physics_injection_full_scan.md))*

### 机制(回答"为什么全废"):load-bearing problem

物理 slot 只占 **2/192 维、~1% 梯度**;而**黑盒 190 维冗余编码了同一份位置**——预测走黑盒旁路即可、绕过 slot(下图 probe-190 三域实测:去掉 slot、只用黑盒 190 维,位置照样解得出、与全 192 维几乎一致)。

![](figures/fig15_bypass_probe190.png)

> **图说明**:三个域,每域两根柱——**黑柱=用全部 192 维解位置**、**蓝柱=去掉物理 slot、只用黑盒 190 维解位置**(probe ρ,越高越可解)。两柱几乎等高(0.78–0.92)=**黑盒单独就把位置编了进去**;底部红带=随机 2 维对照(0.2–0.5,解不出)。→ **位置冗余铺在黑盒里、预测可绕过任何物理 slot(旁路的直接实证)**。

**可证伪验证(LBR)**:把 slot 加权承重(pw1→300),危害确实随之响应——但**4 域×分区只有 2 个回到持平、从不净增益**,冲量域(collision)任何权重都救不回。证明"机制方向对、但旁路在、修不了根本"。

![](figures/fig8_lbr_ablation.png)

> **图说明**:左 = uniform 上 pos_weight 从 1→300 的曲线(蓝 both-OOD / 橙 r/m-OOD,虚线带 = 各自 baseline);右 = 三域的 nMSE/baseline 比值随 pos_weight。→ 加权只把 **uniform·both 拉回持平**(而 r/m-OOD 全程救不回),**collision 任何权重都在 baseline 之上、且越加越差** → 承重是"机制可证伪验证",不是修复方法(2/4 判决格回持平、从不净增益)。

---

## 2. 发现二(正,支配变量):同一代码下,训练协议 2.2–8.3×

**① free-rollout(受控析因变量 + 阳性对照,不 claim 方法新颖性)**。只把 teacher-forcing 换成自回归 free-rollout 这**一个开关**(修 exposure bias、不灌任何物理):合成三域 **2.2–3.6×**、照片级仿真 Physion++ **8.3×**(均三种子、区间零重叠),**每个 OOD 分区连 ID 都提升**。→ 它同时**堵死"你实现有 bug/指标失效"**:同一套代码、同一批指标下别的干预能大赢,矛头钉死在结构本身。

![](figures/fig2_free_rollout.png)

> **图说明**:每域两根柱 = teacher-forced(原始)vs free-rollout(我们)的 rollout 误差(nMSE↓),柱顶数字 = 下降倍数。合成三域 + 照片级仿真 Physion++ 全部大幅下降,**仿真数据反而更猛(8.3×)**:

| 域 | teacher-forced | free-rollout | 倍数 |
|---|---|---|---|
| uniform | 0.300 | 0.136 | 2.2× |
| parabola(r/m) | 0.443 | 0.122 | 3.6× |
| collision | 1.153 | 0.479 | 2.4× |
| **Physion++(照片级仿真,h64)** | 1.174 | 0.141 | **8.3×** |

**② rollout horizon 匹配动力学复杂度**:碰撞吃长 rollout、光滑域不吃;Physion++ 顶配 np28+scale 把 h64 nMSE 打到基线的 **1/19**(0.280→0.014)、无拐点。

---

## 3. 评测口径(Setup 级,一页带过——不是主线发现)

判决指标 = **nMSE + pixel PSNR**(外部指标、所有案例同向);**cos/probe 只当诊断**——它们是训练目标的对偶量,加对应 loss 必涨,当主指标会高估物理结构(实测反转:cos 升 1.50× 而真值 nMSE 崩到 0.21;我们早期 sweep 亦被带偏后翻案)。parabola 判决走 r/m-OOD(nMSE 除零爆点)。**论文里放 Setup 一段 + 附录,不单开章。**

![](figures/fig5_cos_trap.png)

> **图说明**(备询问用):三案例中每例 **蓝柱(cos)>1 说"变好"、红柱(pixel/nMSE)<1 说"变差"** → 只看 cos 结论相反;判决必须 nMSE/pixel。详 → [detail/evaluation_traps.md](detail/evaluation_traps.md)。

---

## 4. 结论:不是物理没用,是"共享 latent 嫁接"没用

我们否定的是**往共享 latent 嫁接物理状态**(冗余、被旁路)。要让预测**天然依赖**物理态,得改**架构**(extrinsic:低维物理态=预测唯一必经通道)。我们把官方 PIWM 忠实移植验证:它学到正确物理、ID/v-OOD 甚至比 LeWM 更准,**但 size/mass-OOD 仍崩**(ρ 0.33 vs 0.89)——**extrinsic 解决"承重",但不解决"编码器-OOD",是必要非充分**。

![](figures/fig9_piwm_vs_lewm.png)

> **图说明**:官方 PIWM(紫,extrinsic 架构)vs LeWM free-rollout(蓝)的 rolled-out 位置 ρ(↑),分 4 个 OOD 分区、两个域。**PIWM 学到正确物理、ID/v-OOD 甚至更准,但阴影的 size/mass-OOD 崩**(编码器扛不住没见过的球尺寸):

| 分区 | PIWM | LeWM |
|---|---|---|
| ID | **0.96** | 0.93 |
| **r/m-OOD** | 0.33 ⚠️ | **0.89** |
| v-OOD | **0.97** | 0.87 |
| **both-OOD** | 0.48 ⚠️ | **0.87** |

> → extrinsic 架构解决了"承重/旁路",但**没解决"编码器扛不住 OOD"**——是通向物理一致的必要条件、非充分条件。

**⏳ 跨 backbone 泛化(DINO-WM,另一 session 在跑、回来填)**:同一协议(TF vs FR + structpos 臂)搬到 DINO-WM——填入后"LeWM 上的发现"升级为"JEPA 系 latent WM 的共性";若不同向则如实写为边界条件。

---

## 贡献一览(可当 slide 尾页)

| # | 贡献 | 一句话 |
|---|---|---|
| 1 | **系统解剖 + 机制** | 30 格全扫:注入 29/30 不优于 baseline;根因 = latent 已冗余编码状态 + 黑盒旁路(*decodable but not load-bearing*),probe-190/LBR 可证伪验证 |
| 2 | **支配变量 = 训练协议** | 同一代码同一指标下 free-rollout 2.2–8.3× + horizon 匹配 1/19 → 短板在长程动力学、不在状态 |
| 3 | **extrinsic 归因** | PIWM 忠实移植:学到正确物理但 r/m-OOD 崩(0.33 vs 0.89)——必要非充分;⏳ DINO-WM 跨 backbone 复核待填 |
| 4 | **评测协议**(Setup 级) | 判决用 nMSE/pixel,cos/probe 只诊断(训练目标对偶量) |

**为什么能投 AAAI**:不靠"新方法"(free-rollout = scheduled sampling,主动引用;它是**析因变量与阳性对照**),靠**一个反直觉、有机制、跨机制×域×基准系统验证的科学发现**——诚实、完整、有解释力。

---

*图矢量版(插 PPT)在 [figures/](figures/) 的 `.pdf`;一键重画 [figures/storyline_figures.py](figures/storyline_figures.py)。数据源表见 [detail/figures_gallery.md](detail/figures_gallery.md)。*
