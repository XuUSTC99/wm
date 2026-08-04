# ICLR GIPP 实验设计

更新时间：2026-08-04（Asia/Shanghai）

## 核心假设

对长时程视觉世界模型，如果在每一步 rollout 后，从冻结 latent 中线性解码物理状态、进行物理积分，再将状态修正写回预测 latent，可以降低长程误差。写回采用经验 latent Mahalanobis 度量下的最小修正，因此对任意可逆线性 latent 换基保持等变。

## 协议修复

PhyWorld 是被动视频数据。主实验统一使用 `wm.use_action=false`、常量 action 列，并跳过 action normalization。旧版真实/合成 action 结果仅作为 privileged upper bound，不进入主表。

## 分阶段实验矩阵

1. 单元测试与冒烟测试：验证投影后状态命中物理目标、换基等变性、`alpha=0` 完全恢复黑盒 predictor，以及两批次训练可运行。
2. 止损实验（uniform）：比较 action-free free-rollout baseline 与 GIPP `alpha={0.25,0.5,1}`，评估 h1/h8/h16/h28。
3. 平滑动力学：在 parabola 上比较 constant-velocity 与 gravity 目标。
4. 碰撞：先做 free-flight 投影与保守 alpha；只有平滑域止损通过后，才加入接触/事件 gate。
5. 控制实验：随机 decoder、打乱状态目标、错误重力、`alpha=0`、普通伪逆对比协方差投影、仅末步投影对比每步写回。
6. 稳健性：种子 3072/1234/42，四个 ID/OOD 分区，LeWM 与冻结 DINOv2，最后扩展到 Physion++ h64/h128。

## 成功与止损标准

- oracle GIPP 若在 h16/h28 latent nMSE 至少改善 10%，且 h1/h8 退化不超过 5%，则继续。
- 主结果门槛：至少两个物理域、两个 backbone 上，h64/h128 改善不少于 15%。
- 若随机或 shuffled 控制仍保留收益，立即停止当前叙事并重新设计。

## 产物目录

- 权重与转换后数据：`/data1/likun-share/junjxu/.stable_worldmodel/`
- 日志、配置与指标：`runs/iclr_gipp/<实验名>/`
- 冻结 GIPP decoder：`runs/iclr_gipp/state_decoders/`
