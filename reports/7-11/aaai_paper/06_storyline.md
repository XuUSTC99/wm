# 论文故事线 —— Toward Physics-Consistent Latent World Models

**对应稿件**:[paper/main.pdf](paper/main.pdf)(AAAI-27,正文 6 页 + 附录,已编译)
**日期**:2026-07-12
**用途**:讲给导师/合作者、写 rebuttal、检验叙事是否闭环时随时调。

---

## 一句话主线

> **物理信息"存在"于世界模型的 latent 里(可解码),但预测从不"使用"它(不承重)——所以往共享 latent 上嫁接物理结构全都白费甚至有害;真正让模型遵守物理的是训练协议,不是结构先验。**

核心概念 ***decodable but not load-bearing*(存在 ≠ 使用,presence ≠ use)**:decodable = 信息在场,load-bearing = 预测真的靠它。全文发现都挂在这个落差上。

**更精确的机制表述(信息论+建设性,2026-07-13 收敛)**:为什么"嫁接白费甚至有害"?——因为 **latent 已经把物理"状态"编强了(位置 ρ 0.9、冗余分布在黑盒 190 维,probe-190 实证)**,再注入**同一份状态**是**冗余的**(不带新信号 → 不提升)、还占表示容量分梯度(→ 有害,与预测目标打架)。**真正的短板不在"状态",而在"预测器的长程动力学"**(状态可解码但 rollout 会漂)——这一句就把"注入物理状态没用"和"修训练协议(free-rollout)有用"统一了。物理注入若要帮,前提是补 latent **真正缺的东西**(动力学/守恒量),而非已在里面的状态(⚠️此点未证实,唯一疑似正例 parabola 速度 −0.026 不倚重,future work)。这也**免疫"换更高维 slot / 更好映射会不会好"的质疑**:那还是塞冗余状态、没堵旁路;要突破须改架构堵旁路(extrinsic),不是换编码。详见 [detail/why_physics_structure_fails.md 层3](detail/why_physics_structure_fails.md)。

---

## 逻辑链(每一步承接上一步)

1. **现象**:JEPA 世界模型能把物体位置编得极准(probe ρ 0.96)、单步预测近乎完美(cos 0.98–0.99),但一自回归 rollout 就崩(collision h28 cos 掉到 **0.24**,而同时位置一直可解码 ρ≈0.84),OOD 更崩。→ **状态可解码,不代表物理可遵守。**（📊 [Fig 1](figures/fig1_thesis_presence_not_use.png)）

2. **现状**:学界两条药方——① 注入物理结构(PIWM / 深监督);② 堆数据(PhyWorld 已证 video 生成这条不行)。**但"物理结构能不能救 latent 世界模型"从没被系统测过。**

3. **缺口**:既有正面证据几乎都建在 cos/probe 这类指标上——而这些是**训练目标的对偶量**,加了对应 loss 必然涨,不代表预测变好;而且注入方式、初始化、域都是零散测的,从没和"训练协议"这个最大混淆变量对比过。→ **一个基本的析因问题没人回答:训练做对之后,物理结构还有用吗?**

4. **我们的做法**:一次把设计空间扫满——**5 种注入机制 × 2 种训练方案 × 3 个物理域 × 2 个照片级仿真基准**,种子受控,>60 次训练。

5. **发现一(负,论文主体)**:**物理结构不是通用杠杆**。把各种物理量(位置/速度/加速度)固定编码进 slot,30 个"机制×域"格子里**对整体数据几乎都是损害或持平**;唯一例外是 parabola 上把速度也编进 slot 有**一点点**提升(r/m 0.122→0.096,量级很小),**可能**因为速度在抛体里是随时间线性变化的驱动量、比二次位置好外推——但这点小提升不构成可用方法(同一编码在 uniform/collision 上反而变差 +0.071/+0.142)。从头共训比后训练嫁接**伤得更狠**(uniform Δ 从 +0.035 放大到 +0.558),堵死"要在预训练注入才行"的辩护。（📊 [Fig 3](figures/fig3_physics_signflip.png)）

6. **机制(回答"为什么全废")**:**物理 slot 占比低、被预测绕过(load-bearing problem)**([论据详见 detail/load_bearing_reweighting.md](detail/load_bearing_reweighting.md))——物理 slot 只占 2/192 维、~1% 梯度,黑盒 190 维还冗余编码了位置,预测绕开 slot 走黑盒;物理梯度和预测梯度打架(比值 15–125×)。**可证伪验证(LBR 全曲线消融,pw1→300)**:加权到头,4 个域×分区只有 2 个救回持平(uniform·both、parabola 高权 r/m),uniform·r/m 和 collision 全程救不回、collision 还越加越差——**加权只在 slot 占主导那格消掉危害,多数格子仍有害、无净增益**,证明了机制方向对但修不了根本(旁路在)。**这一步把"负结果"变成"有机制解释的科学发现"。**(📊 [Fig 8 LBR 边界条件](figures/fig8_lbr_ablation.png))

7. **发现二(正,支配变量)**:真正**跨域不翻车**的杠杆都在训练和数据侧,有三个:① **free-rollout**(uniform/parabola/collision 2.2–3.6× + 真实 Physion++ **8.3×**,唯一跨合成/仿真都通用的主升力;📊 [Fig 2](figures/fig2_free_rollout.png),论据 [detail/free_rollout_evidence.md](detail/free_rollout_evidence.md));② **rollout horizon 匹配动力学复杂度**(碰撞吃长 rollout、光滑域不吃;真实数据 np8→28 长程 nMSE 单调降到 1/19、无拐点;📊 [Fig 7](figures/fig7_realdata_num_preds.png));③ **几何 scale 增广作长程稳定器**(和长 horizon 协同,仿真上拿到 19×,且不反转)。**增广必须拆开看,不能整体算正面**:appearance 增广只是**简单合成域专属**的最强杠杆(−48~63%),一到照片级仿真就**反转 100×**(因为 appearance 在照片级场景里携带物理信息——摩擦/质量/材质,不是可抹掉的 nuisance)——所以 appearance 是**反面边界警示,不是正面方法**(📊 [Fig 4](figures/fig4_aug_synthetic_vs_real.png),论据 [detail/augmentation_synthetic_vs_real.md](detail/augmentation_synthetic_vs_real.md))。正面框架 = 发现支配变量(free-rollout / horizon / scale),而非负结果堆;appearance 的反转本身也是一条贡献(合成→仿真的边界)。

8. **发现三(方法论)**:四个评测陷阱——解释了为什么既有文献"看起来"物理结构有效(cos/probe 对偶陷阱:cos 升 1.50× 而真值崩到 0.21)、为什么合成→仿真的边界被忽略、zero-shot 迁移封顶=random 先验(0.607,无配置能超)。**顺手回收了别人被骗的原因。**(📊 [Fig 5 cos 陷阱](figures/fig5_cos_trap.png)、[Fig 6 迁移天花板](figures/fig6_transfer_ceiling.png),论据 [detail/evaluation_traps.md](detail/evaluation_traps.md)、[detail/real_data_physion.md](detail/real_data_physion.md))

9. **结论(建设性,不自我否定)**:我们没否定物理结构本身,只否定"往共享 latent 上**嫁接**"。**extrinsic 架构**(低维物理态是预测唯一必经通道)才让预测天然依赖物理态——**我们把官方 PIWM 忠实移植到 phyworld 验证了这点:它学到正确物理、ID/v-OOD 比 LeWM 还准,但 size/mass-OOD 崩(ρ 0.33 vs 0.89)——PIWM 的红利来自它的架构而非方程,而其 VAE 编码器同样扛不住 OOD**。这既解释了别人的正结果、又和我们的负结果自洽,还指出了唯一出路(future work)。(📊 [Fig 9 PIWM 对照](figures/fig9_piwm_vs_lewm.png),论据 [detail/../piwm_baseline/PLAN.md](../piwm_baseline/PLAN.md))

---

## 三个发现如何互锁(故事的精髓)

不是三块拼盘,是**一个论证**:

- 发现一说"结构没用" → 立刻有人问"是不是你实现有 bug / 指标不对"
- 发现二(训练协议大赢)证明**同一套代码、同一批指标下别的干预能大幅提升** → 排除"实现/指标失效",反衬出是结构本身没用
- 发现三(评测陷阱)解释**为什么别人以为结构有用** → 补上"那前人的正结果哪来的"这个缺口
- 机制(占比低+被旁路绕过)+ LBR 验证把这三者**统一**到一个可证伪的解释下

关于 parabola 那点小提升(−0.026 nMSE):**不当卖点、不当证据支柱**。它量级很小、只在一个域一个分区出现、可能含随机成分——诚实说法就是"物理量进 slot 整体有害,parabola 偶尔小赢一点,可能因速度在抛体里可外推"。反驳"实现有 bug"不靠它(靠 probe-190 旁路 + structured_loss 降 + slot 可解码涨 + 梯度打架 + free-rollout 阳性对照,见下)。

### 最致命的质疑:"发现一结构没用是不是你实现有 bug"

负结果最容易被这样打发——"你物理 loss 没真接进去,所以才看不出效果"。反驳的关键是**不辩"检查过代码"(苍白),而用模型的可观测行为证明约束真生效了**,只是生效方向无益/有害(①bug 没接上→和 baseline 无异;②真生效→被改变但方向有害;证据全指向②)。行为证据 + free-rollout 阳性对照:structured_loss 在降、slot 可解码性大涨、**probe-190 实测黑盒旁路存在**、按 pos_weight 甜点系统响应、梯度层面与预测拔河、危害随强度单调。

> 六证据的**逐条数字、剂量-反应曲线、"编码方式次优"升级质疑的反驳**(全带数据与出处)→ **[detail/why_physics_structure_fails.md](detail/why_physics_structure_fails.md)**。

---

## 配图（PPT 用）

每张图的**数据表 + 原始数据源**见 **[detail/figures_gallery.md](detail/figures_gallery.md)**;矢量 `.pdf`（插 PPT）在 [figures/](figures/),脚本 [figures/storyline_figures.py](figures/storyline_figures.py) 一键重画。**时间紧只讲前 3 张**（钩子→正面→负面+机制）足以立住主线。

**Fig 1 — 可解码≠预测依赖它（钩子）**
![](figures/fig1_thesis_presence_not_use.png)

**Fig 2 — free-rollout 跨域通用 2.2–8.3×（正面地基）**
![](figures/fig2_free_rollout.png)

**Fig 3 — 物理结构只在匹配动力学时帮（负结果+机制）**
![](figures/fig3_physics_signflip.png)

**Fig 4 — LBR 边界条件（可证伪验证）**
![](figures/fig8_lbr_ablation.png)

**Fig 5 — PIWM 外部 baseline 也崩 OOD（归因架构/编码器）**
![](figures/fig9_piwm_vs_lewm.png)

**Fig 6 — 增广合成→真实反转 100×**
![](figures/fig4_aug_synthetic_vs_real.png)

**Fig 7 — cos 陷阱（方法论）**
![](figures/fig5_cos_trap.png)

**Fig 8 — zero-shot 迁移天花板：没配置超 random 先验**
![](figures/fig6_transfer_ceiling.png)

**Fig 9 — 真实数据：长 rollout 单调好、无拐点**
![](figures/fig7_realdata_num_preds.png)

---

## 每节在故事里的角色

| 节 | 角色 |
|---|---|
| Intro | 抛出"可解码≠遵守" + 缺口(析因没人做) |
| Related | 定位三条线(基准 / 物理结构 / exposure bias),点名将被检验的对象 |
| Setup | 交代受控实验设计(为什么 1K、为什么单机制、指标为什么用 nMSE/pixel) |
| §4 物理失效(心脏) | 发现一 + 机制(占比低+被旁路绕过) + LBR 可证伪验证 |
| §5 什么有效 | 发现二(训练/数据支配变量 + 合成→仿真边界) |
| §6 评测陷阱 | 发现三(回收"别人为什么被骗") |
| §7 结论 | 归因架构、指出 extrinsic 出路、诚实 limitation |

---

## 为什么这个故事能投 AAAI

不靠"新方法"(free-rollout = Scheduled Sampling,增广有竞品,都不新),靠**一个反直觉、有机制、跨机制×域×数据集系统验证的科学发现**:

> *物理归纳偏置在共享 latent 世界模型上系统性失效,因为信息可解码但预测不依赖它;真正的杠杆在训练协议。*

加上评测方法论(cos 陷阱)和对 PIWM 的架构性归因,这是一篇诚实、完整、有解释力的实证论文——审稿人挑不出"证据不足"或"叙事不闭环"。详细创新性评估与审稿人反对意见预案见 [02_story_and_novelty.md](02_story_and_novelty.md);数字总账见 [01_results_ledger.md](01_results_ledger.md)。
