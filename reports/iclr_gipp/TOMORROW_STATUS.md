# 当前实验状态

更新时间：2026-08-06 21:55（Asia/Shanghai）

## 一句话结论

零训练 Shadow-State Writeback alpha=0.75 是当前唯一跨三个随机种子稳定改善 h16 与 h28、同时保持 h1 不变的配置。加入 5 epoch correction-aware 微调后结果反而不稳定，因此训练版已否决，零训练版保留为 ICLR 主方法候选。

## 已完成实验

- action-free 匀速、抛体与碰撞基线；
- 固定输出投影 alpha=0.25/0.5/1.0；
- innovation 门控与 horizon 门控；
- Shadow 写回 alpha=0.5/0.65/0.75/1.0；
- alpha=0.75 三种子 5 epoch 训练感知版本；
- 三种子相同训练预算继续训练对照；
- 每个最终训练实验均完成 500 条轨迹评测。

## 当前可信结果

零训练 Shadow alpha=0.75：

| 指标 | 基线均值 | Shadow 均值 | 相对改善 |
|---|---:|---:|---:|
| h1 | 0.0148 | 0.0148 | 0.0% |
| h16 | 0.6735 | 0.6043 | 10.3% |
| h28 | 1.5517 | 1.4746 | 5.0% |

训练感知 Shadow 相对等训练量对照：

| 指标 | 对照均值 | Shadow 均值 | 相对变化 |
|---|---:|---:|---:|
| h16 | 0.6605 | 0.7451 | +12.8% 退化 |
| h28 | 1.5404 | 1.5155 | -1.6% 改善但种子不一致 |
| both-OOD | 0.7708 | 0.8008 | +3.9% 退化 |

## 文件入口

- 详细人工判决：`reports/iclr_gipp/RESULTS.md`
- 自动全量指标：`reports/iclr_gipp/AUTO_RESULTS.md`
- 机器可读指标：`reports/iclr_gipp/metrics.csv`
- 方法与创新性：`reports/iclr_gipp/NOVELTY_AND_METHOD.md`
- 实现说明：`reports/iclr_gipp/IMPLEMENTATION.md`
- 原始评测日志：`runs/iclr_gipp/eval/`
- 训练日志：`runs/iclr_gipp/finetune/`
- 模型权重：`/data1/likun-share/junjxu/.stable_worldmodel/iclr_gipp/finetune/`

所有结果均来自远程主机，未在本地执行或保存训练产物。
