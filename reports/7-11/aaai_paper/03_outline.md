# 论文骨架(AAAI-27,五章版 2026-07-16;与 [06_storyline.md](06_storyline.md) 逐节对应)

标题:**Physics Is Already There: Rethinking Physical Inductive Biases in Latent World Models**

主张编号 C1~C7 对应 [01_results_ledger.md](01_results_ledger.md)。**全文五章;§3(方法)与 §4(实验)一一对应;指标讨论压进 §4.0 Setup 一段,不单开章。**

---

## Abstract(~180 词)

结构:动机(物理注入在 latent WM 上没被系统检验)→ 现象钩子(状态可解码、rollout 却违反物理)→ 做了什么(5 家族 10 臂 × 3 域 + 2 照片级仿真基准 + 协议受控对照,~200 run)→ 发现 1(注入 29/30 不优于 baseline,从头共训更伤)→ 机制(latent 已冗余编码状态,注入=塞冗余+抢梯度,预测走黑盒旁路;LBR 可证伪验证)→ 对照落点(训练协议 2.2–8.3×,短板在长程动力学不在状态)→ extrinsic 必要非充分(PIWM 移植)。核心概念:*decodable but not load-bearing*。⚠️ 动机不提 free-rollout;评测陷阱不进 abstract。

## 1. Introduction(~1 页,动机单线;蓝本 = storyline §C1)

| 段 | 内容 | 硬数据/图 |
|---|---|---|
| P1 现象钩子 | latent WM 单步近乎完美(cos 0.99)、状态可解码(probe ρ 0.80–0.96 both-OOD),但 rollout 违反动力学(collision h28 cos 0.24) | Fig.1 teaser |
| P2 现状 | 物理注入在别的形态成功(PIWM extrinsic 专用架构;deep-sup 低维状态 latent),**在共享高维 latent WM 上零散、无系统检验** | — |
| P3 概念先定义 | latent WM/JEPA、rollout、teacher forcing、OOD 分区、nMSE/PSNR(导师反馈:别让读者带着未定义概念读) | — |
| P4 核心问题(唯一动机) | **物理注入能否让 latent world model 物理一致?** 我们扫满设计空间来回答 | — |
| P5 我们做什么 | 5 家族 10 臂 × 3 域 × 2 训练方案 + 2 照片级基准;机制假设+可证伪检验;协议受控对照;外部 extrinsic 对照(+DINO-WM 跨 backbone ⏳) | — |
| P6 答案预览 | 29/30 不优于 baseline、a=g 也伤、从头共训 Δ+0.558;机制=冗余+旁路;协议 2.2–8.3×;extrinsic 必要非充分 | — |
| P7 贡献四条 | ①系统解剖+机制(decodable≠load-bearing) ②可证伪验证(probe-190/LBR) ③支配变量=训练协议(析因) ④extrinsic 归因(PIWM,必要非充分)+跨 backbone(⏳) | — |

⚠️ **free-rollout 不出现在动机段**——首次出场在 §3.3,身份 = 受控析因变量 + 阳性对照。

## 2. Related Work(半页~2/3 页,三条线;蓝本 = storyline §C2)

物理注入 WM(PIWM 2412.12870/2503.02143、deep-sup 2504.03861——如实承认其成功与适用域);latent WM 与物理评测基准(PhyWorld/Physion/Physion++,一律"photorealistic simulation");rollout 训练与 exposure bias(Bengio 2015/Ranzato 2016,主动承认 free-rollout 不新)。

## 3. Method:Physics Injection Design Space(~1.25 页;蓝本 = storyline §C3)

- **3.1 注入设计空间**:5 家族 → 10 臂,**每族写设计目的**(表见 storyline §3.1);三正交轴闭合(硬/软 × 状态/演化 × 有/无标签)+ 从头共训 vs 后训练 2×2。
- **3.2 机制假设与可证伪检验设计**:load-bearing 假设(2/192 维、~1% 梯度、黑盒冗余);probe-190(旁路实证)与 LBR pos_weight 全曲线(剂量-反应)两个检验的设计逻辑。
- **3.3 训练协议对照**:free-rollout 的两重身份(析因变量 + 阳性对照)+ exposure bias 分析 + horizon 匹配。**不 claim 方法新颖性。**
- **3.4 外部对照设计**:PIWM 忠实移植(extrinsic);DINO-WM 复制(跨 backbone ⏳)。

## 4. Experiments(~2.25 页;蓝本 = storyline §C4;**小节与 §3 一一对应**)

- **4.0 Setup(~0.75 页)**:LeWM 架构/init/种子;三域 + OOD 四分区;Physion++ 直训 + Physion OCP。**指标一段带过**:判决 = nMSE + pixel PSNR(外部、同向);cos/probe 仅诊断(训练目标对偶量,反转实锤一句 + Fig 5 置附录);parabola 判决走 r/m-OOD(除零爆点)。详细指标论证 → detail/evaluation_traps.md,正文不展开。
- **4.1 (↔3.1) 注入全扫**:Tab.2 30 格 + Fig.16 热力图;25 差/4 平/1 赢;a=g 也伤;2×2 从头更伤;posvel·parabola 例外 = 机制签名;迁移旁证(pos_weight 0.551 < random 0.607);**Physion++ 上 struct/cons 同样 3–10× 差于 FR**。
- **4.2 (↔3.2) 机制**:probe-190(0.78–0.92 vs 随机 2 维 0.2–0.5)+ 加权 loss 比 15–125× + PR 塌方 41→4 + LBR 全曲线(2/4 格回持平、无净增益、collision 越加越差)。Fig.15 + Fig.8。
- **4.3 (↔3.3) 协议对照**:三域 2.2–3.6× + Physion++ 8.3×(三种子零重叠;全 4 分区含 ID 2.0–4.6×);horizon np8→28 h64 1/19 无拐点。Fig.2 + Fig.7。
- **4.4 (↔3.4) 外部对照与泛化**:PIWM 学到正确物理但 r/m-OOD 崩(ρ 0.33 vs 0.89)→ extrinsic 必要非充分,Fig.9;**DINO-WM 数据位 ⏳**(TF vs FR / structpos 一臂 / presence,填法见 storyline §4.4)。
- **4.5 小结一段**:同一 latent 同一代码同一指标——注入已有状态无效有害、修长程动力学大赢、extrinsic 必要非充分 → *decodable but not load-bearing*。

## 5. Conclusion(~0.5 页;蓝本 = storyline §C5)

结论 + 建设性出路(注入 latent 真缺的东西 / extrinsic+鲁棒编码器)+ Limitations(单 backbone——DINO-WM 填入后弱化;Physion++ 物理臂单种子但差距 3–10×;照片级仿真非真实视频;26/30 单种子的辩护;posvel 例外未系统探索)。

## 图表清单(7 图 2 表 + 附录)

| # | 内容 | 论文位置 | 数据来源 |
|---|---|---|---|
| Fig.1 | teaser:decodable ≠ compliant | §1 | fig1(aaai_p0 probe/rollout) |
| Fig.16→论文 Fig.2 | 30 格热力图(主图) | §4.1 | fig16(structdyn_eval + aaai_p0) |
| Fig.15→Fig.3 | probe-190 旁路实证 | §4.2 | fig15 |
| Fig.8→Fig.4 | LBR 剂量-反应 | §4.2 | fig8 |
| Fig.2→Fig.5 | TF vs FR 四域(+DINO-WM 待填) | §4.3 | fig2 |
| Fig.7→Fig.6 | num_preds by-horizon | §4.3 | fig7 |
| Fig.9→Fig.7 | PIWM vs LeWM 四分区 | §4.4 | fig9 |
| Tab.1 | 30 格全表(判决分区 nMSE) | §4.1 | full_scan §3 |
| Tab.2 | Physion++ 直训(TF/FR/物理臂 by-horizon) | §4.1/4.3 | physionpp logs |
| 附录 Fig | cos 陷阱三案例(fig5)+ 迁移天花板(fig6) | Appendix | fig5/fig6 |

## 写作规则

- 数字全部从 [01_results_ledger.md](01_results_ledger.md) 引,每个数字标注 `raw_data/` 源,写作时跑 citation-audit。
- Physion/Physion++ 一律 "photorealistic simulation",禁 real-world/real video。
- 标题、正文不出现 "PIWM"/"Physically Interpretable World Models" 之外的未定义缩写;概念先定义后使用。
- LaTeX:AAAI-27 模板;图用 matplotlib 出 pdf([figures/storyline_figures.py](figures/storyline_figures.py))。
