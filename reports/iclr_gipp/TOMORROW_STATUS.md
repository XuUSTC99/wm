# 明日实验查看入口

更新时间：2026-08-05 01:35（Asia/Shanghai）

## 当前结论

当前最可信的结果是 Shadow-State Writeback，alpha=0.75。它在匀速域三个种子上保持 h1 完全不变，同时使 h16 平均改善 10.3%、h28 平均改善 5.0%，且两个长程节点均为三种子一致改善。OOD 仅作为附加指标，不作为当前主线判据。

## 今晚任务队列

第一优先级是 6 个已启动的训练感知实验：三个 Shadow 微调与三个等训练量对照，全部 5 epoch，结束后自动评测 500 条轨迹。当前使用 GPU 2、4、5、6；GPU 2 与 4 各并行两个任务，以利用 80GB 显存。没有抢占 GPU 0、1、3、7 上已有的其他进程。

第二优先级是在上述任务完成后汇总逐种子 h1、h8、h16、h28、ID 和 OOD 指标，判断收益来自 shadow 训练还是普通继续训练。只有 Shadow 相对等训练量对照仍在 h16/h28 稳定占优，才升级为论文主结果。

第三优先级是补更长预测范围。现有标准轨迹只能可靠评到 h28；将先审计原始 HDF5 是否含更长帧序列。如果数据足够，则补 h64/h128；如果不足，则生成独立长轨迹测试集，绝不把无真值的自回归稳定性当精度。

第四优先级是完成碰撞域基线汇总以及 shadow 适用性测试。抛体域已有直接物理投影失败证据，不在长程主线确认前继续烧大量 GPU。

## 文件位置

- 当前人工结论：reports/iclr_gipp/RESULTS.md
- 自动指标表：reports/iclr_gipp/AUTO_RESULTS.md 与 metrics.csv
- 方法与创新性：reports/iclr_gipp/NOVELTY_AND_METHOD.md
- 实验设计：reports/iclr_gipp/EXPERIMENT_DESIGN.md
- 实际实现：reports/iclr_gipp/IMPLEMENTATION.md
- 训练日志：runs/iclr_gipp/finetune/
- 评测日志：runs/iclr_gipp/eval/
- 模型权重：/data1/likun-share/junjxu/.stable_worldmodel/iclr_gipp/finetune/

## 明日判决规则

- 通过：Shadow 相对等训练量对照在三个种子的 h16、h28 均值上改善，并且至少两个种子同向，h1 基本不变。
- 条件通过：h16 稳定改善但 h28 方差较大；继续做长轨迹与校正强度自适应。
- 否决：收益可被普通继续训练解释，或 h28 只由单个种子贡献。

所有文档使用中文；所有代码、训练和数据操作均在远程主机完成。
