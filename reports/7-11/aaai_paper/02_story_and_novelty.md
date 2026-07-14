# 故事线与创新性评估(AAAI-27)

**先说结论**:推荐 **Story A+B 混合**——"系统解剖 + 最小建设性修复",统一论点是 **"可解码 ≠ 预测依赖它"(presence ≠ use)**。免得读者误会:这不是"负结果堆",而是"找到支配变量(预测是否依赖物理 slot / 表示挤占)并给出边界条件"的实证科学。

---

## 1. 我们手里有什么(资产盘点,按论文价值排序)

| 资产 | 强度 | 单独当创新点行吗 |
|---|---|---|
| 物理结构先验全线失效的**完整解剖**(5 种注入 × 2 init × 3 域 × 合成/真实)+ 机制解释 | ★★★★ | **行**——系统性、有机制、直接对话 PIWM/deep-sup 文献 |
| **评测三陷阱**(cos 对偶陷阱 / 迁移天花板=random 先验 / 协议混淆假阴性) | ★★★★ | 接近——方法论贡献,审稿人喜欢,但单独撑不起主线 |
| 增广**合成→真实反转**(合成最强杠杆,真实崩 100×) | ★★★ | 半行——反直觉发现,但 SPARK 等增广文献已密集 |
| free-rollout 修 teacher forcing(三域+真实通用) | ★★(作为方法 **2/10**,lewm 会话 novelty check 已定) | **不行**——= Scheduled Sampling(Bengio 2015);只能当"训练协议压倒结构先验"论断的证据 |
| pos_weight 加权 niche + 四条件边界 | ★★ | 不行——边际收益;但作为"机制解释的可证伪验证"价值很高 |
| horizon-complexity matching(num_preds 律) | ★★ | 不行——但是干净的实证规律,做小节 |
| Physion++ 直训 np20sc(h64 nMSE 3.2×) | ★★ | 不行——工程性;当真实数据验证章 |

**novelty check 既有结论(lewm 会话 7-08/7-09,勿重复劳动)**:free-rollout 2/10、增广 2/10;竞品 SPARK(2510.24216,物理引导增广)、PIAug(2311.00815)、Augmented World Models(2104.05632);[UNVERIFIED] Persistent Robot WM 2603.25685、Sword 2605.07288、Train-Short-Infer-Long 2602.14027(写作前须 citation audit)。

### 名词表:被解剖的五种物理注入方式(论文 §4 / C4 表的行)

| 注入方式 | 约束什么 | 要位置标签吗 | 一句话原理 |
|---|---|---|---|
| structpos 固定编码 | 状态是什么 | 要 | 强制 latent 前 2 维 = 真实位置(`emb[:,0:2]≈proprio`),无读出头 |
| probe 深监督 | 状态可读出 | 要 | 额外线性头从 latent 读出物理量(`probe_head(emb)≈proprio`),latent 无需固定维对齐 |
| 运动学头 dynamics | 演化(固定方程形式) | 要 | 位置 slot 按 `z+v+a` 显式外推;a 可为 0(匀速)/g(严格 PIWM 重力)/MLP(自由) |
| **consistency loss** | **演化(无固定形式)** | **要** | 预测 rollout 的位置 slot 做差分得"预测速度",强制等于真值速度(可加二阶);不假设 a 的形式 → 专为 collision 冲量设计(smooth-a 必死处) |
| **无标签物理先验(label-free)** | **演化(只要求形式)** | **不要** | 不钉真值,只要求 slot"必须按二阶动力学平滑演化"(即 `z_{t+1}=z_t+v_t+a_t`,序列的二阶差分≈加速度必须小/连续,不能跳变;匀速 a=0、抛体 a=g 满足,碰撞冲量不满足),指望位置信息自组织进 slot;动机 = physion_collide 纯视频无 proprio,一切带标签监督失效。对照组 grounded = 同结构 + 钉真值 |

五行从"钉状态"扫到"钉演化"、从"有标签"扫到"无标签",设计空间闭合;全负(含 grounded)→ 支撑"根因是架构(物理 slot 占比低+被黑盒旁路绕过),不是方程形式或标签有无"。

**四把尺子(所有数字的指标定义)**:

| 尺子 | 定义 | 量什么 | 坑(对应 C6) |
|---|---|---|---|
| cos | 预测 latent 与真值 latent 的余弦 `ẑ·z/(\|ẑ\|\|z\|)` | 方向一致性 | 尺度盲:真值(1,0) 预测(5,0) 也得满分;幅度过冲/崩塌看不见(friction cos 0.99 而 nMSE 24.6) |
| probe-ρ | 线性头从 latent 解码物理量,与真值的 Pearson 相关(K=4 = 堆 4 帧,速度可差分) | 信息**存在**(可读出) | 是 probe/structured 训练目标的对偶量,加了该 loss 必然涨,不代表预测变好 |
| **nMSE** | `‖预测−真值‖²/真值方差`,0=完美、≈1=瞎猜均值 | 预测真误差(方向+尺度) | 分母退化除零爆炸(parabola h28 球出框→方差→0→nMSE 飙百万);先查 by-horizon 再引用 |
| **pixel PSNR** | 预测 latent 解码成图 vs 真实帧,`10·log10(MAX²/MSE)` dB,+3dB≈像素误差减半 | 端到端画面(位置权重天然高) | 依赖 decoder(collision decoder-limited,只用 latent 尺) |

上两行量"存在"、下两行量"使用";判决规则 = 逐分区、逐 horizon、nMSE/pixel 为主、双指标交叉验证。

**核心概念:intrinsic vs extrinsic(物理状态住在哪)** —— *intrinsic*(我们/LeWM+slot):物理是共享 192 维 latent 里切出的 2 维"租客",共享黑盒 predictor 滚整体,190 维黑盒冗余编码位置 → **有旁路,slot 可被绕过,预测天生不依赖它**。*extrinsic*(PIWM):独立低维物理态 z_p **就是**主状态,由固定形式动力学方程演化,decoder 被强制只从 z_p 重建 → **无旁路,架构结构上强制预测必经它**(+分阶段训练避免梯度打架)。我们把 PIWM 的零件(方程形式/slot/probe/from-scratch/有无标签)逐一嫁接到 intrinsic 上全部失败 → **PIWM 的红利归因于 extrinsic 架构整体,而非物理方程知识**;整换 extrinsic 是设计空间仅剩的未测分支(P2-1 spike / future work)。

**核心概念:存在 ≠ 预测依赖(presence ≠ use)** —— 区分两件事:①位置信息**存在**于 latent(可解码,probe ρ 0.96 ✅);②预测**真正依赖**这几维(它们错则预测错、loss 疼 ❌)。物理 slot 满足①不满足②:pred_loss 对 192 维平均后这 2 维只占 ~1% 梯度,且 190 维黑盒**冗余编码了位置**(predictor 走黑盒旁路即可,不必经过 slot),外加物理梯度与预测梯度打架(比值 15–125×)→ 预测不依赖它。**可证伪验证**:pos_weight=30 提高 slot 在 loss 里的占比 → structpos 危害消失(0.183→0.132 持平);但加大占比并没消除黑盒旁路,一切挂在被旁路绕过的 slot 上的方程/约束仍全灭。PIWM 之所以行,是 extrinsic 架构让低维物理态**就是**主 latent(预测唯一必经通道)。此落差 = 我们命名的核心概念 *decodable but not load-bearing*(presence ≠ use;不再作标题,升级为正文机制短语,主标题见 §标题候选 #0)。

## 2. 三条候选故事线

### Story A:系统解剖(推荐主线)
> **"物理归纳偏置没有让 latent world model 更懂物理——训练协议才有。我们解释为什么。"**

- **主张 1(反直觉核心)**:在共享 latent 的 JEPA 世界模型上,物理结构先验(固定 slot / 运动学方程 / 深监督 probe / consistency / 无标签先验)**系统性失败**——无论嫁接还是 from-scratch、有无标签、软硬约束、正确与否的物理形式(严格重力形式也伤)。机制:**预测不依赖物理通道**(2/192 维被平均 loss 稀释,黑盒通道冗余编码位置、预测走会漂的黑盒路)+ 梯度冲突(probe/pred 梯度比 15–125×,intrinsic dim 塌方)。
- **主张 2(建设性)**:真正有效的是训练与数据侧:free-rollout(唯一跨合成/真实通用)、rollout horizon 匹配动力学复杂度、域匹配增广(附反转警示)。
- **主张 3(方法论)**:评测三陷阱——先前文献看到"物理结构有效"很可能是 cos/probe-ρ 这类**监督对偶量**造成的假象(我们多处用 nMSE/pixel 反转)。
- **优点**:数据全在手、诚实、有攻击性(直接对话 PIWM/2504.03861)、三主张互锁成完整故事。
- **风险**:AAAI 评审偏好新方法;需要用"发现支配变量"的正面框架而非"负结果"框架来写。

### Story B:机制的可证伪验证(并入 A,不单独成篇)
> 把 pos_weight 加权(LBR)当作机制验证:加大物理 slot 在 loss 里的占比,看危害是否随之消失。
- **诚实口径(全曲线 pw1→300)**:加权到头,4 个域×分区只有 2 个救回与 baseline 持平(uniform·both、parabola 高权 r/m),另 2 个全程有害(uniform·r/m、collision,后者越加越差)。→ 加权**只在 slot 占主导那格消掉危害,多数格子仍有害、无净增益**。这反而让 Story A 更统一——即使加权到极致,物理结构整体仍是负效果,验证了"机制方向对但修不了根本(旁路在)"。**别把 LBR 写成"修复/最小修复"**(它只 2/4 格回持平),写成"机制的可证伪验证"。
- 作用:仍是机制解释的可证伪验证("按机制修,危害确实消失");不能当 headline。

### Story C:诊断+治疗(extrinsic 架构)——本轮放弃
> 解剖之后真的换 extrinsic 架构(独立低维物理 latent + 对抗解耦 + 分阶段)修复。
- **10 天造不出来+验证**;且 grounded 都失败,风险极高。留作 ICLR-27(9 月截稿)的升级路线或 rebuttal 弹药。idea-stage 的 PhysConsist-Rollout(守恒流形投影)同此定位。

## 3. 推荐定位(一段话,可直接改成 abstract 骨架)

> Latent world models can *decode* physical state, yet fail to *obey* physical law over long-horizon rollout and under OOD physics. We systematically inject physical inductive biases into a JEPA-style world model (LeWM) across five mechanisms (fixed slots, kinematic dynamics, deep-supervision probes, consistency losses, label-free priors), two injection regimes (post-hoc / from-scratch), three synthetic physics domains, and two real-world video benchmarks — and find they consistently *hurt*. We trace the failure to a **load-bearing problem**: physical dimensions occupy a vanishing fraction of the latent and gradients, so prediction routes around them; a one-line loss reweighting (LBR) that makes the slot load-bearing flips it from harmful to helpful, but only within sharp boundary conditions (smooth dynamics, pixel-space evaluation). What robustly helps instead is the training protocol: autoregressive free-rollout, horizon matched to dynamics complexity, and domain-matched augmentation — though augmentation gains reverse catastrophically from synthetic to real data. Finally we show why prior evidence for physics priors may be illusory: cosine/probe metrics are *duals of the training loss* and systematically mislead. (数字往里填:三域 both-OOD −57~65%、Physion++ h64 3.2×、cos→nMSE 100× 反转。)

**主标题**:**Toward Physics-Consistent Latent World Models: Why Injecting Physics Doesn't Help, and What Does**
- 备选:*Training Beats Structure: An Anatomy of Physical Robustness in Latent World Models* / *Presence Is Not Use: Physical State Is Decodable but Not Compliant*
- 约束:禁用 "PIWM"/"Physically Interpretable World Models" 字样;不放 LeWM(泛化 scope、避免单-backbone limitation 上标题,LeWM 放 abstract/setup)。

## 4. 相关工作对比(novelty 台账)

| 文献 | 他们说 | 我们的 delta |
|---|---|---|
| PhyWorld(Kang et al. 2024,数据来源) | video-gen(diffusion)靠 scaling 学不会物理规律,case-based 泛化 | 同一问题搬到 **latent/JEPA WM** + 追问"注入物理结构能不能救"→ 不能,并给机制;用他们的 OOD 协议,可比性天然 |
| PIWM(2412.12870)+ Four Principles(2503.02143) | 物理结构化 latent + 固定形式动力学有效(from-scratch + extrinsic) | 在**共享 latent(intrinsic)**设定下系统证伪其可移植性:连严格重力形式、from-scratch 都伤;指出其成功依赖 extrinsic 架构整体而非物理方程本身(我们 pretrain 2×2 排除了"from-scratch 就行"的解释) |
| Deep-supervision probes(2504.03861) | probe-in-loss 提升可解码性+减漂移 | 复现后在可信指标上反转(pixel h28 −0.71dB、both-OOD nMSE +0.036);其表面收益 = cos/ρ 对偶陷阱 |
| Scheduled Sampling / exposure bias(Bengio 2015 起) | 自回归训练要消 exposure bias | 不 claim 方法新;贡献是"**在物理 WM 上它压倒一切结构先验**"这一系统证据 + horizon-复杂度匹配律 |
| SPARK(2510.24216)/ PIAug 等增广 | 增广治 dynamics OOD | 补上"合成→真实反转"的边界:appearance 增广真实数据 nMSE 崩 100×;增广收益是域特定的 |
| Physion / Physion++(评测基准) | 真实物理理解基准 | 提供 zero-shot 天花板=random 先验(0.607)的负结果 + 直训可行的正结果 |

## 5. 审稿人反对意见预案

| 预期攻击 | 应对 |
|---|---|
| "负结果不算贡献" | 框架反转:发现支配变量(预测是否依赖物理 slot),负结果是**对既有正面主张的系统检验**;附 LBR 可证伪验证;引用 "What Matters..." 类分析论文先例 |
| **"你实现有 bug / 物理 loss 没真生效"**(最致命) | **不辩"检查过代码",用行为证明约束真生效**:structured_loss 2.53→0.015、slot 可解码 0.31→0.96、按 pos_weight 甜点响应、恰在匹配域(parabola)三种子生效、梯度比 15–125× 可测、危害随强度单调(+0.035→+0.558);外加 free-rollout 阳性对照证明 pipeline/指标灵敏。**弹药库见 [detail/why_physics_structure_fails.md](detail/why_physics_structure_fails.md)** |
| "free-rollout 就是 scheduled sampling" | 主动承认并引用;我们的 claim 是相对重要性(协议>结构),不是方法 |
| "单一 backbone(LeWM/ViT-tiny),结论会泛化吗" | 老实写进 limitation;5-19 的 7-encoder 横评(DiT-XL/ImageNet/PushT)提供部分外推;呼吁社区在更大 backbone 复验 |
| "没有外部方法 baseline(如 PIWM 原实现)" | **已做(2026-07-13)**:官方 PIWM extrinsic code 忠实移植到 phyworld(δ=0 最有利档),学到正确物理(parabola g_y=−0.028≈真值),ID/v-OOD ρ 比 LeWM 更准,但 **r/m/both-OOD 崩(ρ 0.33/0.48 vs LeWM 0.89/0.87)**——**其 VAE 编码器扛不住尺寸 OOD**;佐证"物理结构买不到 OOD 鲁棒"。⚠️单种子。见 [ledger C4 外部 baseline 块](01_results_ledger.md)。**附带 nuance**:extrinsic 修好承重但不修编码器-OOD → 温和了"extrinsic 是唯一出路",诚实框架 = 两条物理路(intrinsic 旁路/extrinsic 编码器)都不买 OOD 鲁棒 |
| "单种子" | P0 补 headline 3 种子;collision 增广已有 3 种子 |
| "为什么信 nMSE/pixel 不信 cos" | §评测陷阱整节论证:cos 是训练目标对偶量 + 三个实锤反转案例(含 nMSE=47659 时 cos=0.95 的发散案例) |

## 6. 投稿决策

- **首选 AAAI-27**(abstract 7-21 / 全文 7-28):Story A+B,现有数据 + P0 补实验足够成稿。
- **备选 ICLR-27**(约 9 月下旬):若导师认为 AAAI 版单薄,加 extrinsic spike(Story C)升级成"诊断+治疗"再投;AAAI 版可作为压力测试。
- 两条路不冲突:AAAI 被拒的反馈直接喂 ICLR 版。
