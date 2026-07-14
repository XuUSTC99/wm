# 论文骨架(AAAI-27,正文页数以 CFP 为准,按 7 页正文规划)

标题(工作稿):**Decodable but Not Load-Bearing: Why Physical Inductive Biases Fail in Latent World Models**

主张编号 C1~C7 对应 [01_results_ledger.md](01_results_ledger.md)。

---

## Abstract(~180 词)
骨架见 02 §3。结构:现象(可解码≠守规律)→ 做了什么(5 注入 × 2 init × 3 合成域 + 2 真实基准的系统检验)→ 发现 1(结构全线伤)→ 机制(物理 slot 占比低+被旁路绕过)+ 最小修复(LBR,带边界)→ 发现 2(有效的是训练协议;增广合成→真实反转)→ 发现 3(cos 对偶陷阱解释既有文献假象)。

## 1. Introduction(七段八股,吸收 papers/mypaper/introduction.md 的导师反馈)

| 段 | 内容 | 硬数据/图 |
|---|---|---|
| P1 问题 | 世界模型短程合理、状态可解码,但长程滚动偏离动力学、OOD 崩。**第一句概括全段;必须带数据**(如:1-step cos 0.99,h28 掉到 0.24;both-OOD 误差 ×3) | Fig.1 teaser |
| P2 现状 | 两条主流药方:①物理归纳偏置(PIWM 系/deep-sup probe);②scaling+数据(PhyWorld 已证 video-gen 不行)。latent WM 上①是否真有效缺系统检验 | — |
| P3 概念先定义清楚 | latent WM/JEPA、rollout、teacher forcing、OOD 分区、nMSE/PSNR(**导师反馈:别让读者带着未定义概念读下去;"换言之"类词删掉**) | — |
| P4 缺口 | 既有正面证据依赖 cos/probe-ρ 这类**监督对偶量**;注入方式、init、域覆盖零散;无"结构 vs 训练协议"的受控对比 | — |
| P5 我们做什么 | 系统解剖(5×2×3+真实)+ 机制假说(物理 slot 占比低+被旁路绕过)+ 最小修复验证(LBR)+ 协议侧对照(free-rollout/horizon/增广) | — |
| P6 效果 | 结构全伤(严格重力形式也伤);LBR 翻正但窄;协议侧 −57~65%、真实 h64 3.2×;cos 三次反转 | — |
| P7 贡献 | 四条:①解剖+机制 ②LBR+边界条件 ③协议配方+增广反转 ④评测陷阱与修正协议 | — |

## 2. Related Work(半页~2/3 页,导师反馈"related 相当于没写"→ 必须实写)
分四小节:物理评测基准(PhyWorld/Physion/Physion++);物理结构化 WM(PIWM 2412.12870、2503.02143、deep-sup 2504.03861——**点名将被检验**);rollout 训练与 exposure bias(Bengio 2015 起,承认 free-rollout 不新);增广与 OOD(SPARK/PIAug,指出无人报告合成→真实反转)。

## 3. Experimental Setup(~1 页)
- LeWM 架构(5 组件、192-D、SIGReg)+ 两种 init(scratch/PushT)。
- PhyWorld 三域 + OOD 四分区协议(r∈[0.7,1.5]、v∈[1,4] 为 ID);Physion++(直训)与 Physion(zero-shot OCP)。
- **评测协议(本身是贡献的一部分)**:nMSE/pixel-PSNR 为判决指标,cos/probe-ρ 仅作诊断;probe 协议三件套(paper-init、K=4、no-projector,附 5-26 假阴性案例);按 trajectory 切分。

## 4. Physical Inductive Biases Fail(C4+C5,~1.5 页,论文的心脏)
- 4.1 注入方式扫描表(Table 2:5 机制 × 3 域,全红)。五种机制的定义/差异表见 [02 名词表](02_story_and_novelty.md)——正文开头用 2-3 句把"钉状态(structpos/probe) vs 钉演化(dynamics/consistency) vs 无标签(label-free)"的谱系交代清楚(琨哥八股:概念先定义后使用)。
- 4.2 pretrain vs post-hoc 2×2(排除"要从头训"辩护;注明 scratch 基线欠拟合 caveat 或用 P0 补跑的干净版)。
- 4.3 机制:物理 slot 被旁路绕过(2/192 稀释、黑盒冗余编码、梯度比 15–125×、intrinsic-dim 塌方)。Fig.4 机制示意 + 证据面板。
- 4.4 LBR 最小修复:pos_weight 甜点 30 翻正(0.183→0.114);四条件边界(光滑域 ✓ / 冲量域 ✗ / pixel 尺才可见 / 伤迁移)。**框架:机制的可证伪验证,不是 SOTA 方法。**

## 5. What Actually Helps: Training and Data(C1+C2+C3,~1.5 页)
- 5.1 free-rollout:三域 −57~65%(Table 1)+ 真实数据同向;定位为 exposure-bias 修复的受控证据。Fig.2 by-horizon 曲线。
- 5.2 horizon-complexity matching:collision np≈20 vs 光滑域 np8(加长反害);Physion++ np20 中程 4×。Fig.3。
- 5.3 增广:合成域配方与交互矩阵(app×np20 冲突/scale×np20 协同);**合成→真实反转**(friction nMSE 100×)。Fig.6。
- 5.4 真实数据上限验证(C7):np20+scale h64 nMSE 0.087(3.2×);deform 共同短板留 limitation。

## 6. Evaluation Traps(C6,~0.75 页,特色小节)
- cos 是监督对偶量(数学论证一段)+ 三个实锤反转(probe 长程、app 增广 100×、发散时 cos=0.95/nMSE=4.7e4)。
- zero-shot 迁移天花板 = random 架构先验(0.607)。
- 协议混淆假阴性(vx 0.166→0.939)。
- 落点:给社区的评测 checklist(4 条)。

## 7. Discussion & Limitations(~0.5 页)
单 backbone(ViT-tiny;5-19 七 encoder 横评部分外推)、单种子处(标明哪些已 3 种子)、extrinsic 架构未检验(后续工作;PhysConsist-Rollout 方向)、deform 域短板。

## 图表清单(6 图 3 表)

| # | 内容 | 数据来源 |
|---|---|---|
| Fig.1 | teaser:decodable≠compliant(左:probe ρ 高;右:rollout 漂移) | 5-27 rollout + probe |
| Fig.2 | TF vs FR by-horizon,3 域 | C1 |
| Fig.3 | num_preds 甜点随域复杂度移动 | C2 |
| Fig.4 | 旁路机制示意 + 梯度比/塌方证据 | C4 diagnostic |
| Fig.5 | cos-vs-nMSE 反转散点(三案例标注) | C6 |
| Fig.6 | 增广交互矩阵 + 真实反转柱状 | C3 |
| Tab.1 | 主表:三域 both-OOD(TF/FR/np/aug/物理各臂) | C1-C4 |
| Tab.2 | 物理注入解剖(机制×域,含 pretrain 2×2) | C4 |
| Tab.3 | Physion++ 直训 by-horizon | C7 |

## 写作分工建议
- intro/abstract/story:先由本 session 出草稿 → 导师过。
- §3-6 数字全部从 01_results_ledger.md 引,**每个数字标注出处 log/报告**,写作时跑 citation-audit。
- LaTeX:AAAI-27 模板(CFP 下载 AAAI Press 格式);图先用 matplotlib 出 pdf。
