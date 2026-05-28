# ARPredictor Rollout 测试提案

**日期**：2026-05-27
**状态**：proposal，待 user 决定是否实施

---

## 1. 为什么要做

目前 5-26 报告所有的 probe 数据测的都是 **state encoding**：

```
encoder(frame_t) → emb_t → linear/MLP probe → pos_t / vel_t
```

K=4 multi-frame probe 也只是把 4 帧 emb concat 让 probe 做帧间差分，**没有调用 LeWM 的 ARPredictor**。

这意味着报告里 "encoder vy ρ=0.98" 这种结论只能说明：
- ✅ encoder 把当前帧的 vel 编码进 emb 了
- ❌ **不能说明** encoder + predictor 能预测下一帧或外推抛物线

要测 "能不能预测抛物线运动" 必须用 ARPredictor 做 rollout。

---

## 2. 当前 vs proposed 协议

| 维度 | 当前 (state encoding) | proposed (rollout prediction) |
|---|---|---|
| **用 encoder?** | ✅ 只用 encoder | ✅ |
| **用 predictor?** | ❌ 完全没用 | ✅ 调用 `model.predictor` 做自回归 |
| **测的能力** | "emb 里有没有当前帧 pos/vel" | "encoder + predictor 联合能不能预测未来帧" |
| **关键 metric** | 当前帧 pos/vel ρ | 1-step / N-step prediction MSE |
| **ID→OOD 含义** | 单帧表征跨分布是否稳健 | **真正的 world model 外推** |

---

## 3. 实施方案

### 3.1 加载已有 ckpt

LeWM JEPA 模型包含完整 `encoder + predictor`：

```python
model = torch.load(ckpt_path)  # 已训练好的 ID-only FT ckpt
encoder = model.encoder
predictor = model.predictor  # ARPredictor，之前没用过
```

3 个 ckpt 已经在手（5-26 这次 ID-only FT 已经跑完）：
- `/home/qlib/.stable_worldmodel/collision_paperinit_id1k/*epoch_20*.ckpt`
- `/home/qlib/.stable_worldmodel/uniform_paperinit_id1k/*epoch_20*.ckpt`
- `/home/qlib/.stable_worldmodel/parabola_paperinit_id1k/*epoch_20*.ckpt`

### 3.2 Rollout 流程

每条 traj T=32 帧。给前 4 帧作为 context（history_size=3 + 当前帧），然后预测剩 28 帧：

```python
# Encode context frames (no gradient)
emb_0, emb_1, emb_2, emb_3 = [encoder(f) for f in frames[:4]]
emb_history = [emb_0, emb_1, emb_2, emb_3]

predictions = []
for t in range(4, T):
    # predictor 接受过去 history_size 帧 emb + actions
    emb_pred = predictor(
        embeddings=emb_history[-3:],      # 过去 3 帧 (history_size=3)
        actions=actions[t-3:t],            # 过去 3 步 action
    )
    predictions.append(emb_pred)

    # Teacher forcing OR autoregressive?
    # (a) teacher: feed real emb_t back
    emb_history.append(encoder(frames[t]))
    # (b) AR: feed predicted emb_t back
    # emb_history.append(emb_pred)
```

两种模式各跑一遍：
- **Teacher forcing**：每步用真 emb 喂回 → 测 1-step prediction quality
- **Autoregressive**：每步用预测 emb 喂回 → 测累积误差

### 3.3 评估 metric

每个 ckpt + 每条 traj：

| Metric | 含义 |
|---|---|
| `mse_emb_1step` | 1-step prediction `‖emb_pred − emb_real‖²` 均值 |
| `mse_emb_Nstep` | t=4 到 t=31 的 AR rollout 累积 emb 误差 |
| `mse_pos_1step` | 用同一个 K=4 probe 从 emb_pred 解出 pos_t，跟真 pos_t 比 |
| `mse_pos_Nstep` | AR mode 下 28-step rollout 后 pos 误差 |
| `mse_vel_*` | 同上但对速度 |

按 partition 切（ID / r-OOD / v-OOD / both-OOD）。

### 3.4 期待的结果模式（猜测）

| 现象 | 期望 |
|---|---|
| Teacher forcing 1-step：跨 partition 表现 | 应该相对稳定（每步重新校准）|
| AR rollout：ID partition 累积误差 | 缓慢线性增长（最好情况）|
| AR rollout：OOD partition 累积误差 | **可能发散**（如果 predictor 没学会 OOD dynamics）|
| Parabola vy 8-step rollout MSE | 真正的 "能预测抛物线" 测试 |
| Uniform vx 16-step rollout MSE | 真正的 "能预测匀速" 测试 |

---

## 4. 工程量

| 任务 | 时间 |
|---|---|
| 写 rollout 脚本（参考 le-wm 的 `eval.py`）| 1-2 小时 |
| 在 3 个 ID-only ckpt 上跑 rollout × 4 partition × teacher/AR | ~30 min compute |
| 整理 metric 表 + per-partition rollout error curve | 30 min |
| **合计** | ~3 小时 wall-clock |

如果加 frozen baseline 和 leaked-FT 对比，时间翻倍。

---

## 5. 决策点

### 做的理由 ✅
- **真正回答 "encoder + predictor 能不能预测物理"**——这是 LeWM 论文的真核心 claim，不是 probe ρ
- 跟现有 state-encoding 数据形成完整 picture：能编码 vs 能预测
- AR rollout 误差曲线是论文级别的图（横轴 step，纵轴 MSE，按 partition 着色）
- ID→OOD 在 rollout 上的表现差异**通常比 single-frame 大得多**——可能这才是真正能区分 frozen vs FT 的 metric

### 不做的理由 ❌
- 当前 5-26 已经有清晰的 state-encoding 数据，rollout 是另一个论文级别项目
- LeWM `eval.py` 已经实现了 rollout，可以直接复用，但调通 + 跨 3 个 domain 适配仍需调试
- 如果只想验证 "FT 能不能 ID→OOD 泛化"，现有 probe 数据已足够说服

---

## 6. 文件位置

- 当前 ID-only ckpt：`/home/qlib/.stable_worldmodel/{collision,uniform,parabola}_paperinit_id1k/*epoch_20*.ckpt`
- 参考实现：[le-wm/eval.py](../../le-wm/eval.py)（LeWM 论文 eval pipeline）
- LeWM predictor 类定义：[le-wm/jepa.py](../../le-wm/jepa.py)
- 当前 probe 脚本（state encoding，仅用 encoder）：[probe_mlp_mse_pearson.py](../../phyworld/scripts/probe_mlp_mse_pearson.py)

---

## TL;DR

**当前所有数据都只测 encoder，没测 predictor。要回答 "能不能预测抛物线"，需要写一个 rollout 评估脚本（~3 小时工程），用 LeWM 已经训练好的 predictor 在 ID-only FT ckpt 上做自回归预测，按 partition 测 N-step MSE 曲线。是否值得做由 user 决定。**
