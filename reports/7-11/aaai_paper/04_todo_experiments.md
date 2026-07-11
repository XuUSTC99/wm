# 截稿前后实验清单(照论文方向排优先级)

GPU 现况参考:8×80GB,单 run ≈ 19–32GB / 30–90 min(20ep)。日期锚:abstract **7-21**、全文 **7-28**。

---

## P0 —— abstract 前必须(7-12 ~ 7-20)

| # | 实验 | 为什么(对应主张) | 成本 | 状态 |
|---|---|---|---|---|
| P0-1 | **headline 3 种子**:uniform/collision/parabola × {teacher-forced, free-rollout} × seeds{1234,42}(3072 已有)= 12 runs | C1 是全文地基,现为单种子;审稿必问 | 12 × ~35min | **🔵 2026-07-11 17:27 已进队列** `run_p0_queue.sh`(日志 `/data1/.../runs/aaai_p0/`) |
| P0-2 | **probe/structpos 2×2 的 pixel eval 补全**(7-07 collector 被杀未完成) | C4/C5 的 2×2 表需要 pixel 尺闭环 | 复用 7-07 已训 decoder,只跑 rollout | **🔵 17:34 GPU6/7 跑中**(pxroll json 落 structdyn_eval) |
| P0-3 | **干净 from-scratch 基线**(lr 2e-5 重跑 scratch 2×2) | pretrain 2×2 的 scratch 基线欠拟合(pred_loss 震荡 2~9) | 60ep × ~2h | **拆两半**:parabola 4 臂由 lewm 会话 120ep 版接管(pp2_par_*,GPU0-3 跑中);um/col 4 臂(60ep)在我队列尾部;我队列的 pp2_par 重复臂已 skip-marker 无害化 |
| P0-4 | **LBR 甜点 3 种子**(uniform structpos_fr_pw30 × seeds{1234,42}) | C5 的 0.114 是论文唯一正向物理结果,必须稳 | 2 × ~40min | **🔵 已进队列**(P0-1 之后) |
| P0-5 | 冻结故事线 + 标题 + abstract 文本(不跑卡) | 7-21 要交 | 写作 | 骨架已备,待导师过故事线 |

> **✅ P0 实验全部收官(2026-07-11 22:43,18 run 零失败),数字已回填 ledger。** 要点:C1 三种子零重叠坐实;C4 干净基线下物理伤更狠(uniform Δ+0.558);**C5 降级——LBR latent 增益是种子噪声(0.132±0.014 vs 0.136±0.007),幸存正向只剩 pixel/可解码性(单种子)**。撞名警示:`pp2_par_*` 归 lewm 会话(120ep),勿再用该前缀起新 run。
> 由 C5 降级新增的可选项 **P1-5**:若论文想保留"承重+运动学 pixel +1.25dB"作正向 claim,须补 structcv_fr_pw100 两个种子 + pixel eval(~2h);否则按"边际、指标依赖"写进 anatomy,不跑。

## P1 —— 全文前(7-21 ~ 7-27)

| # | 实验 | 为什么 | 成本 |
|---|---|---|---|
| P1-1 | **np28sc 补完**(现停在 epoch16 无 eval)+ eval;必要时补 np24 | C7 的"h64 未见拐点"claim 目前上限只到 np20sc,要么补完要么把 claim 收窄 | 续训 ~5h + eval |
| P1-2 | 6 张图出图脚本(matplotlib→pdf)+ Tab.1-3 数字核账 | 成稿 | 1-2 天写作 |
| P1-3 | **citation audit**:SPARK/PIAug/[UNVERIFIED] 三篇(2603.25685/2605.07288/2602.14027)逐一核真 | novelty check 引文有未核实项,引错=硬伤 | 半天 |
| P1-4 | (可选)uniform 增广 aug05 3 种子 | C3 合成侧最强数字(0.068)加固 | 3 × ~40min |

## P2 —— 投稿后 / ICLR-27 升级路线(8 月起)

| # | 方向 | 定位 |
|---|---|---|
| P2-1 | **extrinsic 架构 spike**(独立低维 z_p + 对抗解耦 + 分阶段;physics_paper_design §3 设计已在) | Story C:若 work,ICLR 版从"解剖"升级为"诊断+治疗";若不 work,反而完成机制解释的最后一块拼图 |
| P2-2 | **Physion++ 真 held-out scene OOD**(训刚体、测形变) | 356c245c 评为"论文说服力最强"的缺口;现 by-scene 不是严格 OOD |
| P2-3 | PhysConsist-Rollout(自监督守恒量 + rollout 投影回守恒流形) | idea-stage 存货;label-free、攻长程漂移,可作独立方法论文 |
| P2-4 | PIWM 原实现在 phyworld 上端到端复现 | 补"无外部 baseline"短板(repo 已在 tree 里) |
| P2-5 | 更大 backbone(V-JEPA/更大 ViT)复验承重结论 | 回应"单 backbone 泛化性"攻击 |
| P2-6 | deform 布料短板(形变感知损失/多物体 proprio) | 真实域 limitation 的后续 |

## 排卡建议(P0)

- GPU 0/1/3 空闲即可开;继续遵守"零 compute 进程 + ≥30GB free 才占卡"的规矩,用 `reports/6-24/run_queue.sh` 排队最稳。
- 全部 setsid 脱钩;每个 run 训完自动接 `rollout_eval_id1k.py`(run_structdyn.sh 已内建)。
- P0-1 的 teacher-forced 臂注意:`wm.free_rollout=false wm.num_preds=1`(TF 路径要求 np=1)。
- 种子开关:`seed=<n>`(hydra override;默认 3072,建议 3072/1234/42)。

## 归档纪律

每个 P0/P1 实验完成后把数字**直接回填** [01_results_ledger.md](01_results_ledger.md) 对应主张下,并标注 ✅✅;不再散落新报告,写作期以 ledger 为唯一数字源。
