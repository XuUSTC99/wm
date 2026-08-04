# GIPP 实际实现说明

更新时间：2026-08-04（Asia/Shanghai）

## 已实现内容

- `le-wm/gipp.py`：冻结仿射状态 decoder，以及协方差增益 `K = Sigma W^T (W Sigma W^T + eps I)^-1`。
- `le-wm/jepa.py`：在黑盒 predictor 之后、预测 latent 写回自回归历史之前执行投影。
- `le-wm/train.py`：显式 action-free predictor 协议、常量 action 绕过，以及由配置加载冻结 `.npz` GIPP 产物。
- `phyworld/scripts/fit_gipp_state.py`：在冻结真实 latent 上拟合位置/速度 Ridge decoder，转换为原始 latent 的仿射形式并保存协方差。
- PhyWorld 转换器默认生成常量 action；旧版未来速度/加速度必须显式选择才能启用。
- `rollout_eval_id1k.py` 默认 action-free；`--use-action` 仅用于 privileged legacy 对照。
- 启用 GIPP 时冻结 representation projector；实验启动时同时冻结 encoder，防止 decoder 所依赖的 latent 坐标系漂移。

## 配置示例

```yaml
wm:
  use_action: false
gipp:
  enabled: true
  state_path: /path/to/state_decoder.npz
  alpha: 0.5
  eps: 1.0e-4
  physics: constant_velocity
```

## 可复现性要求

每次启动必须在 `runs/iclr_gipp/` 对应目录保存 stdout/stderr、解析后的 Hydra 配置、Git 状态与提交、CUDA 卡映射，以及评估 JSON。



## 2026-08-05：创新量门控扩展

固定 α 会在黑盒预测本来正确时也持续改写 latent，容易损害 ID 与短程性能。现已加入参数 gipp.gate=innovation：

- 用冻结状态读出器计算黑盒预测与物理一步积分之间的残差；
- 按训练状态各维标准差归一化，避免位置与速度量纲主导关系失真；
- 用平滑阈值门控校正强度，小残差近似保持原预测，大残差才逐渐启用物理运输；
- gate=constant 完全保持首轮 GIPP 行为，作为普通固定投影对照。

状态尺度随冻结解码器写入 npz，不反向更新 encoder/projector。新增门控数值单测后，GIPP 测试为 3/3 通过。

该门控目前只解决平滑动力学中的过度校正；碰撞仍需独立的事件模式选择，不能用恒速投影强行覆盖冲量。
