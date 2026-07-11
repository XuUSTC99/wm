# AAAI-27 论文作战指挥部

**日期**:2026-07-11
**目标**:把 5-12 以来的全部 LeWM × PhyWorld × Physion 工作整理成一篇可投 AAAI-27 的论文。

## ⏰ 硬时间线(2026-07-11 查证)

| 节点 | 日期 | 距今 |
|---|---|---|
| **AAAI-27 abstract 截稿** | **2026-07-21** | **10 天** |
| **AAAI-27 全文截稿** | **2026-07-28** | **17 天** |
| 会议 | 2027-02-16~23, Montréal | — |

来源:[AAAI-27 官网](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)、[deadline tracker](https://www.getpaperpilot.com/deadlines/aaai-2027.html)。
**含义**:没有时间做大的新方法开发。论文主体 = 已有数据 + 少量补实验(种子、pixel 补全)。新方法(extrinsic 等)只能作为可选 spike 或 follow-up。

## 📁 文件索引

| 文件 | 内容 |
|---|---|
| [01_results_ledger.md](01_results_ledger.md) | 全部实验数字按"论文主张"归类的总账(带出处与可信度标注) |
| [02_story_and_novelty.md](02_story_and_novelty.md) | 三条候选故事线、创新性评估(vs PhyWorld/PIWM/deep-sup 等)、推荐定位、审稿人反对意见预案 |
| [03_outline.md](03_outline.md) | AAAI 论文逐节骨架(含 intro 八股、表/图与主张的映射) |
| [04_todo_experiments.md](04_todo_experiments.md) | 截稿前必补实验(P0/P1)与投稿后可做(P2),带卡时估算 |

## 一句话战略

**放弃"物理结构提升性能"的旧故事(已被 5 种注入方式 × 2 种 init × 3 域 + 真实数据全线证伪),转投"系统性解剖"论文**:以"**可解码 ≠ 承重**(presence ≠ use)"为统一论点,讲清(1)什么真的有效(free-rollout / horizon 匹配 / 域匹配增广),(2)物理结构先验为什么失效(表示挤占 + 梯度冲突,机制有证据),(3)评测怎么骗人(cos 陷阱 / 迁移天花板 / 协议混淆)。这是诚实、完整、有机制解释的实证科学论文,数据量足以支撑 AAAI。

## 文件索引(补充)

| 文件 | 内容 |
|---|---|
| [05_draft_abstract_intro.md](05_draft_abstract_intro.md) | Abstract + 七段 intro 英文第一稿(种子定版数字,待导师过) |
| [NOTE_from_lewm_pretrain_caveat.md](NOTE_from_lewm_pretrain_caveat.md) | lewm 会话交接:parabola both-OOD 爆点 caveat + pretrain 2×2 干净数字 |
| [run_p0_queue.sh](run_p0_queue.sh) | P0 实验队列脚本(已跑完,可复用作 P1) |

## 决策日志

- 2026-07-11:确认 AAAI-27 截稿;建此目录;采纳"系统解剖"主线(见 02 的论证);pretrain_physics 2×2 当日出结果(物理 from-scratch 也伤)——负结果闭环完成,故事线不再有翻案风险。
- 2026-07-11 深夜:**P0 实验全部收官**(18 run 零失败)。三个定版:①C1 headline 三种子零重叠(uniform 2.2× / parabola 3.6× / collision 2.4×);②C5 降级——LBR latent 增益是种子噪声,口径改"无害共存";③C4 干净基线下物理伤更狠(uniform Δ+0.035→+0.558)。
- 2026-07-11 深夜:采纳 lewm 交接 caveat——**parabola 一切 both-OOD nMSE 弃用为判决指标**(h28 除零爆点,六臂全中),判决走 r/m-OOD;由此 C6 新增第 4 个评测陷阱(nMSE 分母爆点,与 cos 陷阱方向相反,合成"双指标逐分区逐 horizon 交叉验证"的方法论)。05 草稿同步修正("8.7×"作废)。
