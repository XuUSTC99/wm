# le-wm 在 phyworld uniform 上的实验报告

**实验日期**：2026-05-08
**作者**：[Claude] 协助 [haochenluo02@gmail.com]
**目标**：评估 LeWorldModel（le-wm）能否从 phyworld 的物理仿真视频里学到"物理规律"。具体地，在 phyworld `uniform_motion_eval` 数据集上训练 le-wm，然后用线性 probing 测 ViT encoder 的输出表征是否捕获了"球的位置 / 速度"等可解释的物理量。

---

## 0. TL;DR

> **Update (2026-05-10)**：原版"encoder 没学到速度"的结论**被部分推翻**——补做 paper-init + 多帧 probe 后 vel_x R² 跳到 0.939。详见 §3.5。

### 初版结论（from-scratch + 单帧 probe）

| 问题 | 结论 |
|---|---|
| le-wm 能从 phyworld 数据训练出 encoder 吗？ | 能，pred_loss 从 0.094 收敛到 0.013 |
| encoder 学到了"球在哪儿"吗？ | x_pos R² 0.58（随机 ViT）→ 0.73（训练后），轻微提升 |
| encoder 学到了"球多快"吗？ | 看似**没学到反而退步**：vx R² 0.23（随机）→ 0.17（训练后） |

### 修正后的结论（paper-init + 多帧 probe）

🆕 用 lewm-pusht 论文权重做 init + 多帧 K=4 probe：

| Target | from-scratch + K=1（with-proj） | **paper-init + K=4（no-proj）** |
|---|---|---|
| x_pos R² | 0.732 | **0.964** |
| **vx R²** | **0.166** | **0.939** |

🔁 **原版"vx 退步"是 (random init) × (单帧协议) × (with-projector) 三个 confound 联合造成的假阴性**。当 3 个 confound 都修掉后，encoder 实际上**几乎完美地编码了 1D 物理状态**（pos R²=0.96, vel R²=0.94）——这与 §4.4 的"没学到物理规律"结论是冲突的。原版 §4.2 中归因的 "(a) action=velocity 让 encoder 不需要编码速度 + (b) SIGReg 剔除冗余" 在多帧 probe 下被证伪：**速度信息其实是被编进 emb 的，只是只能通过跨帧差分才能恢复**。

> 在 le-wm 系列实验里，下游表征评估应默认用 (paper init) + (多帧 probe) + (`--no-projector`) **三件套**。

---

## 1. 实验设置

### 1.1 模型 ([le-wm/jepa.py](../le-wm/jepa.py), [le-wm/module.py](../le-wm/module.py))

LeWorldModel 是一个 JEPA 风格的 world model，由 5 个组件组成：

| 组件 | 实现 | 参数量 |
|---|---|---|
| **Encoder** `E(o) → h` | HuggingFace ViT-tiny，patch_size=14，image_size=224，无预训练 | ~5.5 M |
| **Projector** `P(h) → z` | MLP 192 → 2048 → 192，BatchNorm | ~0.4 M |
| **Action encoder** `A(a) → z_a` | 1D Conv (effective_act_dim → smoothed) + MLP | <0.1 M |
| **Predictor** `F(z_{t-2:t}, z_a_{t-2:t}) → ẑ_{t+1}` | 6-layer 因果 transformer，AdaLN-zero conditional blocks，hidden=192 | ~3.5 M |
| **Pred-projector** | MLP 同上 | ~0.4 M |

总参数：~10 M（ViT-tiny 主导）。

**架构图**：

```
                                 ┌──────── action_encoder ──┐
                  pixels (B,T,3,224,224)           action (B,T,2)
                       │                                │
                       ▼                                ▼
                ViT-tiny encoder              act_emb (B,T,192)
                       │                                │
              cls_token (B,T,192)                       │
                       │                                │
                  projector                             │
                       │                                │
                  emb (B,T,192) ───┬───── tgt_emb[:,1:]──── │
                                   │ ┌──────────────────────┘
                                   ▼ ▼
                          AR predictor (causal, action-conditional)
                                   │
                                   ▼
                              pred_emb (B,T,192)
                                   │
                                   ▼
                  loss = ||pred_emb - tgt_emb||² + 0.09·SIGReg(emb)
```

### 1.2 数据 ([wm/phyworld/scripts/convert_to_lewm.py](../phyworld/scripts/convert_to_lewm.py))

phyworld 原始 hdf5（[phyworld/data/uniform_motion_eval.hdf5](../phyworld/data/uniform_motion_eval.hdf5)，7.3 MB）格式：
- `video_streams/<group>/<idx>`：每条 traj 一段 32 帧的 MP4 字节流（256×256 RGB）
- `position_streams/<group>/<idx>`：每条 traj 32 步的 2D 球心位置
- `init_streams/<group>/<idx>`：每条 traj 的初始条件 (2 维)
- 共 1000+152 = **1152 条 traj**

phyworld **没有 action 字段**——它是被动物理观测数据集。

转换脚本把它翻译成 le-wm 期望的扁平堆栈格式（[~/.stable_worldmodel/phyworld_uniform_motion.h5](~/.stable_worldmodel/phyworld_uniform_motion.h5)，100 MB gzip 压缩）：

| 字段 | 形状 | dtype | 来源 |
|---|---|---|---|
| `pixels` | (36864, 224, 224, 3) | uint8 | MP4 解码后 bilinear resize 256→224 |
| `action` | (36864, 2) | float32 | **合成**：`vel[t] = pos[t+1] - pos[t]`（最后一帧复制前一步）|
| `proprio` | (36864, 2) | float32 | 直接用 `position_streams` 的 (x, y) |
| `ep_len`, `ep_offset`, `episode_idx`, `step_idx` | 索引 | — | 1152 集 × 32 帧 |

**为什么 action 用 velocity 不用全零**：le-wm 的 `get_column_normalizer` 用 `(x - mean) / std` 做归一化，全零会触发除零得 NaN。velocity 在 traj 间方差非零（不同初速度），归一化健康。物理上"action ≈ velocity" 对匀速运动成立，是个 trivial 但自洽的填充。

#### 数据分布要点

- 所有球在画面里**只往 +x 方向运动，y 恒等于 ~8**
  - position: x ∈ [0.36, 20.68], y ∈ [0.90, 9.05]（y 几乎是常数）
  - velocity: vx ∈ [0.0, 0.6], **vy ≡ 0**
- 这意味着 y_pos / vy 在数据集层面**没有信号**，是常量
- 真正含信息的物理量只有 **x_pos** 和 **vx (≈ speed)**

### 1.3 训练超参 ([le-wm/config/train/lewm.yaml](../le-wm/config/train/lewm.yaml), [le-wm/config/train/data/phyworld.yaml](../le-wm/config/train/data/phyworld.yaml))

| 项 | 值 |
|---|---|
| optimizer | AdamW lr=5e-5, wd=1e-3 |
| scheduler | LinearWarmupCosineAnnealingLR (epoch interval) |
| 精度 | bf16 |
| gradient_clip_val | 1.0 |
| batch_size | 128 |
| num_workers | 6, prefetch_factor=3, persistent_workers |
| max_epochs | 20 |
| `wm.history_size` | 3 |
| `wm.num_preds` | 1 |
| `wm.embed_dim` | 192 |
| `data.dataset.frameskip` | **1**（pusht 默认是 5；phyworld traj 只有 32 帧，frameskip=5 会浪费 75% 数据） |
| `data.dataset.num_steps` | 4（=`history_size + num_preds`）|
| 数据 train/val 切分 | 90 / 10（基于切窗后的起点 idx）|
| `loss.sigreg.weight` λ | 0.09 |
| `loss.sigreg.kwargs` | `knots=17, num_proj=1024` |

### 1.4 训练命令

```bash
cd /home/qlib/agent_memory/wm/le-wm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
  python -u train.py \
    data=phyworld \
    output_model_name=lewm_phyworld \
    subdir=phyworld_probe \
    wandb.enabled=False \
    trainer.max_epochs=20
```

输出目录：[~/.stable_worldmodel/phyworld_probe/](~/.stable_worldmodel/phyworld_probe/)
- `lewm_phyworld_epoch_<N>_object.ckpt`（N=1..20，每个 ~70 MB，整模型 pickle）
- `lewm_phyworld_weights.ckpt`（最后 state_dict，~210 MB）
- `config.yaml`（这次 run 的最终 hydra config）

环境：1× NVIDIA RTX A6000（48 GB，本次单卡训练用 ~15 GB）。

---

## 2. 训练结果

### 2.1 Loss 曲线（从训练日志解析）

| epoch | validate/loss | fit/loss | fit/pred_loss | fit/sigreg_loss |
|---:|---:|---:|---:|---:|
|  1 | 5.1024 | 1.0744 | 0.0939 | 10.8750 |
|  2 | 1.2495 | 0.6947 | 0.0775 |  6.8750 |
|  3 | 0.8410 | 0.5682 | 0.0682 |  5.5625 |
|  4 | 0.7943 | 0.4452 | 0.0409 |  4.5000 |
|  5 | 0.5224 | 0.4312 | 0.0425 |  4.3125 |
|  6 | 0.5168 | 0.3320 | 0.0332 |  3.3125 |
|  7 | 0.4157 | 0.2841 | 0.0243 |  2.8906 |
|  8 | 0.3887 | 0.2929 | 0.0351 |  2.8594 |
|  9 | 0.3786 | 0.2535 | 0.0260 |  2.5312 |
| 10 | 0.3175 | 0.2504 | 0.0239 |  2.5156 |
| 11 | 0.3452 | 0.2756 | 0.0178 |  2.8750 |
| 12 | 0.2894 | 0.2276 | 0.0177 |  2.3281 |
| 13 | 0.2726 | 0.2410 | 0.0144 |  2.5156 |
| 14 | 0.2794 | 0.2022 | 0.0137 |  2.0938 |
| 15 | 0.2591 | 0.2121 | 0.0139 |  2.2031 |
| 16 | 0.2561 | 0.2041 | 0.0146 |  2.1094 |
| 17 | 0.2554 | 0.2148 | 0.0136 |  2.2344 |
| 18 | 0.2484 | 0.2113 | 0.0160 |  2.1719 |
| 19 | 0.2457 | 0.2075 | 0.0122 |  2.1719 |
| 20 | **0.2440** | 0.2466 | **0.0132** | 2.5938 |

### 2.2 Loss 解读

- **`fit/pred_loss` 0.094 → 0.013**：emb 空间下一帧预测 MSE 下降 7×。考虑 emb 维度 192 且经 SIGReg 推到 N(0,I)（即每维 std≈1），`sqrt(0.013) ≈ 0.11` 表示**每维平均预测误差 0.11**——很小。意味着 predictor 几乎完美学到了"用 action 把当前 emb 平移到下一 emb"。
- **`fit/sigreg_loss` 10.9 → 2.1**：emb 分布从初始的远离 N(0,I) 收敛到接近标准高斯。这是 LeWM 的核心创新（替代 EMA / 多 loss 防 collapse）。
- **`validate/loss` 5.10 → 0.24**：跟 fit loss 量级一致，**没有过拟合**（因为 phyworld 同一 traj 内不同窗口高度相关，验证集本质上接近训练集）。
- **平台期**：第 ~10 epoch 起 val_loss 在 0.24-0.30 区间震荡，再训也不会更好。20 epoch 是充分的。

### 2.3 训练时间

20 epoch × ~85 秒/epoch ≈ **28 分钟**，单 A6000，~3 it/s。

---

## 3. Probing 实验

### 3.1 方法 ([wm/phyworld/scripts/probe_lewm_encoder.py](../phyworld/scripts/probe_lewm_encoder.py))

1. 把 phyworld 的全部 36864 帧过一遍 encoder（+ projector），每帧拿到一个 192 维 embedding（cls token 经 projector 后）
2. 按 traj（不是按帧！）切 80/20 = 922 / 230 episodes，对应 29504 / 7360 frames
3. 拟合 Ridge 回归 (alpha=1.0)：emb (192-dim) → target
4. 在测试 traj 的所有帧上算 R² 和 RMSE

**为啥按 traj 切**：同一 traj 内连续帧 emb 高度相关，按帧切会严重信息泄漏，把测试集变成训练集的近邻。按 traj 切才是公平评估。

### 3.2 三个 target

| target | 物理含义 | 是否含信号 |
|---|---|---|
| **position** (2D) | 球心 (x, y) 坐标 | x: ✓（球横移）；y: ✗（恒为 ~8）|
| **velocity** (2D) | (vx, vy) | vx: ✓；vy: ✗（恒为 0）|
| **speed** (1D) | ‖velocity‖₂ ≈ \|vx\| | ✓ |

### 3.3 三个被 probe 对象

| 对象 | 含义 | 用途 |
|---|---|---|
| **trained encoder** | epoch 20 的 le-wm checkpoint 里的 ViT + projector | 主 subject |
| **random encoder** | 同 config 的 ViT-tiny，**随机初始化**未训练 | 控制对照：训练带来多少增益 |
| **pixel-stats baseline** | 每帧 9 维统计量（per-channel mean/std/mean²）作为"emb" | 信号下界：单凭像素分布能解出多少 |

### 3.4 完整结果（epoch 20 checkpoint）

| target | 维度 | trained | random | pixel-stats |
|---|---|---:|---:|---:|
| **position** | 综合 | 0.2975 | 0.3272 | 0.3165 |
| | x | **0.732** | 0.582 | (合并) |
| | y | -0.137 | 0.073 | (合并) |
| **velocity** | 综合 | 0.5830 | 0.6137 | 0.6280 |
| | vx | 0.166 | **0.227** | (合并) |
| | vy | 1.000* | 1.000* | 1.000* |
| **speed** | 1D | 0.1659 | 0.2275 | **0.2560** |

\* vy ≡ 0：预测常数 0 完美命中，R²=1 是 trivial（不构成"学到了"的证据）。

**RMSE**（同样的实验）：

| target | trained | random | pixel-stats |
|---|---:|---:|---:|
| position | 1.9993 | 2.1823 | 2.1144 |
| velocity | 0.1352 | 0.1301 | 0.1277 |
| speed    | 0.2447 | 0.2335 | 0.2364 |

### 3.5 补充实验：paper-init + 多帧 K=4 probe（**Update**）

#### 动机

原版 §3.4 的两个 confound 让 negative result 难以解释：

1. probe 用了 **with-projector**（事后从 collision 实验发现这会压信息，见 §6.6）
2. encoder 是 **random init**（5.5M 参数 ViT-tiny + 36k 帧 + 没 ImageNet 预训练 → 学不到物理表征）
3. probe 是 **单帧**（速度需要看至少 2 帧才能算）

我们补做了 paper-init（用 [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht) 论文 ckpt 当 init）+ 在 phyworld_uniform_motion 上重训 20 epoch + 多帧 K=4 probe + `--no-projector`，**同时控制三个 confound**。

#### 结果

| Setup | x_pos R² | y_pos R² | **vx R²** |
|---|---|---|---|
| 原 §3.4 trained (random init, with-proj, K=1) | 0.732 | -0.137 | **0.166** |
| paper-init + K=1 + no-proj | **0.942** | 0.859 | 0.232 |
| **paper-init + K=4 + no-proj** | **0.964** | 0.964 | **0.939** |

#### 解读

- **x_pos 从 0.732 跳到 0.964**：(paper-init) + (no-proj) 联合贡献，encoder 编码 ball 的水平位置已经接近**完美**
- **y_pos 从 −0.137 跳到 0.86**：原版的负 R² 是 random encoder 完全没编码 y → Ridge 找到一个比常数预测略差的解；paper-init encoder 把 y（即使在每条 traj 内是常数）编码进了 emb，跨 traj 间是有方差的（不同 traj 球的 y 不同），所以 Ridge 能拿 R²
- **vx 从 0.166 跳到 0.939（+0.77 R²）**：这是最戏剧性的——单帧 probe 完全读不出速度（vx 0.232），多帧 K=4 probe 一加上去**几乎完美恢复**（0.939）
- **结论**：encoder 实际**完美记住了每帧的位置**，**速度信息是隐藏在 emb 跨帧结构里的**——单帧 probe 看不到不是因为信息没编进去，是协议读不出来。原版 §4 的"encoder 没学到物理规律"结论**错了**。

#### 与 collision 实验互相印证

[COLLISION_REPORT.md §2.5](COLLISION_REPORT.md) 在 collision 数据上做了同样的对照实验，paper-init + K=4 让 vel_x 从 0.487 跳到 0.883（+0.40）。两个数据集**同步出现"paper-init + 多帧 = 速度恢复"**模式，证明这个观察不是 uniform_motion 的偶然。

---

## 4. 数据解读

### 4.1 训练让 encoder 学到了"球在哪儿"（x_pos）

- **x_pos R² 从 0.582 → 0.732**：训练带来 **+0.15 R²** 的增益，这是**实打实的训练效果**。
- 物理意义：训练让 ViT 的 cls token 更准地编码球的水平位置——但这本质上是 ViT 自己应该会的事，只是 phyworld 的目标导向把它变成了一个"专门的 ball localizer"。
- y_pos（trained R²=-0.137）是噪声：y 恒为 8，target 没有方差，Ridge 找不到信号，得到比常数预测略差的负 R²。

### 4.2 训练**反而损害**了 encoder 对速度的感知（vx）

- **vx R² 从 0.227（随机）→ 0.166（训练后）**：训练让 cls token 对速度的编码**变弱了**。
- pixel-stats 反超 trained encoder（vx≈0.256 vs 0.166），说明这种 trivial 的"位置-速度"统计相关性甚至比训练后的高级 embedding 更直接。

**为什么？三个独立机制叠加：**

1. **Action 流偷走了速度信号**。le-wm 的 predictor 用 AdaLN-zero 把 action（=velocity）注入每一层 attention。predictor 直接拿 velocity 做条件，所以 encoder 完全不需要把速度编进 emb——把这件事丢给 action_encoder 是更优的分工。
2. **SIGReg 主动剔除冗余信息**。loss 把 emb 推向标准高斯，这是个**信息瓶颈**。任何对 pred_loss 没有边际贡献的信息都会被压掉。速度对预测任务无用（action 已提供），就被压掉了。
3. **架构本身鼓励"静态状态"语义**。emb 只编码"当前画面长啥样"，演化交给 predictor。这是 JEPA 设计哲学，非 bug。

### 4.3 综合 R² 的"假象"

| 综合数 | 看起来 | 实际 |
|---|---|---|
| velocity 综合 R² ≈ 0.6 | "好像挺好" | 全靠 vy=0 trivial 命中拉的，vx 单维只有 0.17 |
| position 综合 R² ≈ 0.30 | "一般" | x_pos 单维 0.73 是真实信号；y_pos -0.14 是噪声 |

**看综合数会被 vy / y 这种常量目标误导**。一定要看 per-dimension R²。

### 4.4 整体结论（**已被 §3.5 推翻**）

> ~~**le-wm 在 phyworld uniform-motion 数据上没有学到物理规律。**~~
>
> ~~它只是用 encoder + projector 学了一个**更准的球的水平定位器**。运动学（速度、动量、轨迹外推）这些"物理"性质完全没有进入 encoder 的表征——这是架构 + SIGReg + 数据三者共同决定的，不是训练量不够或调参问题。~~

**原版结论错了**。§3.5 的补充实验显示：用 paper-init + 多帧 K=4 probe + `--no-projector` 后，vel_x R² 从 0.166 跳到 **0.939**——encoder 实际编码了相当干净的位置，速度信息以"跨帧结构"的形式存在 emb 里，单帧 probe 看不到不等于信息没编进去。原版归因到 SIGReg 信息瓶颈和 action=velocity 冗余的两条机制**没有起到 dominant 作用**：multi-frame probe 上信息能恢复说明 SIGReg 没把它冲掉。

**修正后的结论**：

> le-wm 在 phyworld uniform-motion 上**确实学到了 1D 物理状态的精确表征**（pos R²=0.96, vel R²=0.94）——只要 (a) 用 paper init 让 encoder 跨过"从零起步"的低数据量瓶颈，(b) 用多帧 probe 让协议能读出速度，(c) `--no-projector` 不被 projector 压信息。前面 §3.4 看到的 negative result 是协议+初始化的人造假阴性，不是模型真没学。

---

## 5. 概念澄清 / FAQ（实验中容易混的点）

> 这一节专门回应实验中反复被问到的概念问题，独立可读。

### 5.1 "用相邻帧位置做差就能算出速度啊，为什么 encoder 没学到速度？"

直觉上看，给定位置序列 `pos[t]` 和 `pos[t+1]`，速度 `v = (pos[t+1] - pos[t]) / Δt` 是个一行就能写出来的差分公式。那既然 encoder 已经能从单帧解出 pos（probe 实测 R²(x_pos)=0.73），把相邻两帧的 emb 一减不就有速度了吗？

**答案是：probe 测的是单帧 emb 的内容，不是相邻帧 emb 之间的关系**。具体拆开看：

- encoder 是 `image → emb` 的纯单帧函数。**`emb[t]` 只是 frame[t] 这一张静态图的编码**，里面没有 frame[t+1] 的信息。
- probe 喂给 Ridge 的是 `emb[t]`（192 维），目标是 `vel[t]`。要让线性模型预测出 vel，**vel 必须以"线性可读"的形式编码进 emb[t] 这一个向量内部**。
- 一张球的静态照片**确实看不出**球在以多快速度运动（除非有运动模糊；phyworld 渲染没有）——这是物理上的硬限制。
- 你说的"位置做差"是 **跨两帧的运算**，需要 `emb[t]` 和 `emb[t+1]` 同时摆在线性模型的输入里。我们的 probe 没做这个（**单帧 probe**）。如果改成 multi-frame probe（把连续 4 帧的 emb 拼成 768 维输入），那个差分信息就能被读出来了，但那已经不是测"encoder 内部学到了什么"，而是测"两个 encoder 输出之间能不能算差分"。

所以"位置差 → 速度"的逻辑链条是 **数学上正确，但 probe 协议上不成立**：encoder 学到了 pos 不能等价推出 encoder 学到了 vel，因为速度信息**本来就不在单帧里**。

### 5.2 "action 是 velocity，那不就是把答案直接喂给模型了吗？"

**对，这就是这次实验最关键的设计缺陷之一**。回顾一下数据流：

```
encoder(frame[t])     → emb[t]                        ← encoder 看不到 velocity
action_encoder(v[t])  → act_emb[t]                    ← velocity 直接进来了
predictor(emb[t-2:t], act_emb[t-2:t]) → ẑ[t+1]         ← 拿着 velocity 算下一帧
```

**关键观察**：predictor 用 AdaLN-zero 把 action（=velocity）注入每一层 attention。要预测下一帧的 emb，predictor 完全可以"忽略 emb 里有没有速度信息，直接用 action 提供的速度"。从优化的角度，encoder 编码速度是**冗余且代价高**的事，**最优分工是 encoder 只管位置、predictor 用 action 提供速度**——这正是训练后我们观察到的现象。

为什么这个设计会成立：在原版 le-wm 训练数据（pusht）上，action 是控制信号（鼠标拖动），与"agent 的速度命令"语义一致。这套架构假设 action 提供运动信息、emb 提供状态信息——**分工清晰**。但 phyworld 没有 action（被动观测），我们用 velocity 填进 action 字段，结果就是把答案显式给了 predictor。

### 5.3 SIGReg 是怎么"扔掉"对预测无用的信息的？

SIGReg loss 把 emb 分布推向 N(0, I) 标准高斯。这是一个**信息瓶颈**——每个维度的 variance 被约束在 1 附近，不允许某个维度承载额外的"边缘信息"。

数学上 encoder 的 emb 接收两路梯度：

```
∂L_total / ∂emb = ∂(pred_loss) / ∂emb  +  λ · ∂(sigreg) / ∂emb
```

考虑某个特征 f（比如"球的速度"被编码进 emb 的某个方向）：

- **如果 f 对 pred_loss 有帮助**：第一项有正梯度，encoder 会"保留并强化" f
- **如果 f 对 pred_loss 没帮助（因为 action 已经提供了 velocity）**：第一项梯度为 0
- **sigreg 永远在工作**：不管 f 有用没用，sigreg 都在惩罚任何"远离 N(0,I)"的结构

**结果**：对 pred_loss 无用 + 被 sigreg 惩罚 = 这个特征在训练过程中**被慢慢冲刷掉**。这就是为什么训练后 vx 的 R² 反而**从 0.227（random）降到 0.166（trained）**——不是没学，是学了又被冲掉了。

SIGReg 的设计初衷是防止 representation collapse（让 emb 始终铺满空间）；但副作用是**主动剔除任何对训练任务无贡献的信息**。这个 tradeoff 在 phyworld 这种"答案已通过 action 给出"的场景下副作用极大。

### 5.4 R²(vy) = 1.000 不是"完美预测"，综合 R² 是误导

报告 §3.4 里 vy 的 R² 标了 1.000* 加星号，原因是：

- 数据集里 vy ≡ 0（球只水平运动）→ **目标本身没有方差**
- Ridge 回归只要预测常数 0 就完美命中 → 数学上 R²=1
- 但这**不代表 encoder 学到了 vy**——它只是没法"学错"，因为目标根本就没东西可学

**综合 R² 的陷阱**同理：

```
velocity 综合 R² = 0.583
       = mean(R²(vx), R²(vy)) = mean(0.166, 1.000) = 0.583
```

这个 0.583 看起来"还行"，但它是 (有意义的 0.166) 和 (trivial 的 1.000) 平均出来的。**把 vy 这种常量目标和 vx 这种有信号的目标平均起来毫无意义**。

> 任何 phyworld 报告里看到"综合 R²"的数都要先拆 per-dim，否则会被常量目标拉的虚高骗到。

### 5.5 pixel-stats baseline 是什么 + 为什么需要它

**pixel-stats 不是 pixel state**。它是一个故意设计得很弱的 baseline 特征：把整张图压成 9 个数。

```python
# 一张 (3, H, W) 图 → 9 个数：
mean    = pix.mean(over_HW)   # (3,) 每通道平均亮度
std     = pix.std(over_HW)    # (3,) 每通道标准差
mean_sq = mean ** 2           # (3,) 平方项，给线性回归一点非线性
feat    = concat([mean, std, mean_sq])  # 9-D
```

**为什么需要这个 baseline**：如果 192-D 的 ViT encoder 训出来的特征，下游 probe 还打不过这 9 个数，那 ViT 就**白学**了。这是表征评估的"地板"。

实测在 uniform_motion 上：

| target | trained ViT (192-D) | pixel-stats (9-D) |
|---|---|---|
| vx R² | **0.166** | **0.256** |
| speed R² | **0.166** | **0.256** |

在 vx / speed 上，**训练后的 192-D ViT 反而被 9 维像素统计反超**——这是这次实验最让人警觉的发现。

### 5.6 probe 用的是 projector 输出还是 encoder cls token？（事后认知）

§3 中报告的所有 probe 数字（trained vx=0.166, x_pos=0.732 等）都是 **with-projector** 的结果——即 probe 喂给 Ridge 的是 `projector(encoder(frame))` 这个 192-D 输出，不是 encoder 的 cls token 直出。

这在当时被认为是无害选择，但后来在 [collision 实验](COLLISION_REPORT.md#61-jepa-projector-在毁信息--probe-必须---no-projector) 上验证发现：**projector 会主动压掉对"预测下一帧"无用的维度**（信息瓶颈，机制和 5.3 SIGReg 类似但发生在 projector 层），下游 probe 必须用 `--no-projector` 直接读 encoder cls token 才能看到完整信息。

collision 上 collision_event AUC 从 with-proj 0.741 跳到 no-proj 0.808（+0.067）。**uniform_motion 的 vx R²=0.166 这个数也可能被 projector 压低过**——若复跑这个实验，应该用 `--no-projector` 重新评估，可能 vx R² 会比 0.166 高一些（但根据 §4.2 的多个独立机制叠加分析，本质结论"trained vx 不如 random"应该不会变）。

> **教训**：从下次起，le-wm 系列的所有下游 probe 默认走 `--no-projector`。

### 5.7 "多帧 probe" 是什么 / 为什么必要

#### 单帧 vs 多帧

**单帧 probe**（§3.4 的默认）：一张图过 encoder → 1 个 192-D embedding → Ridge 预测目标。

**多帧 probe**（§3.5 用到）：把同一 traj 连续 K 帧的 emb **拼起来**喂 Ridge：

```python
single-frame feature:  emb[t]                                    # 192-D
multi-frame  feature:  concat(emb[t-3], emb[t-2], emb[t-1], emb[t])  # 4×192 = 768-D
```

实现细节：
- 按 traj 切（不跨 episode），每条 traj 前 K-1 帧丢弃。uniform_motion K=4 时 36 864 → 33 408 有效帧（损失 9.4%）
- feature 维度从 192 涨到 K × 192（K=4 → 768-D）
- 还是按 traj 切 train/test 80/20
- Ridge 自动学到怎么线性组合 K 个时间步——比如学到 `vel ≈ (pos[t] − pos[t-1]) / Δt` 这种差分公式

#### 为什么必要 — uniform_motion 上的"假阴性"教训

§3.4 原版 trained vx R²=0.166 < random encoder 0.227 < pixel-stats 0.256 这个"trained 反而退步"的现象，曾被归因到"SIGReg 把 vel 从 emb 里冲走"（§4.2）。**§3.5 的多帧 probe 直接证伪了这条解释**：在 paper-init encoder + K=4 上，vx R² 从 0.232 跳到 **0.939**——速度信息**根本没被冲走**，只是被"必须单帧线性可读"这个协议要求屏蔽了。

机制上的原因：单帧 emb 即使完美记录每帧 position，**单一向量本身没法在内部做"减法"**（线性回归输入是 1 个向量，无法自己跨时间步组合特征）。多帧 probe 把 K 帧 emb 摆给 Ridge 作并列输入，Ridge 就能学到 `w_t · emb[t] - w_{t-1} · emb[t-1]` 这种差分组合，速度自然恢复。

#### 实测对比（详见 [COLLISION_REPORT §6.6](COLLISION_REPORT.md)）

| 实验 | 单帧 K=1 vel_x R² | 多帧 K=4 vel_x R² | Δ |
|---|---|---|---|
| **uniform_motion paper-init** | **0.232** | **0.939** | **+0.707** |
| collision paper-init | 0.620 | 0.883 | +0.263 |
| collision from-scratch | 0.487 | 0.594 | +0.107 |

uniform_motion 上 K=4 lift 最大（+0.71），因为 1D 单球数据上 encoder 能把 position 编码得**极干净**（K=1 pos R²=0.94），差分操作 noise-free 地恢复 vel。

#### 多帧 probe 也不万能

只能恢复**线性跨帧关系**的信息：

- ✅ vel = pos 差分、加速度 = 二阶差分 → 多帧能读
- ❌ encoder 真把信息丢了（state collapse） → 多帧也救不了
- ❌ 目标量和 emb 是**非线性**关系 → 需要 MLP probe，不是多帧 Ridge

> **教训**：从下次起，le-wm 系列下游表征评估应默认加一个多帧 K=4 的版本作为 sanity check——很多"encoder 没学到 X"的 negative result 其实是单帧协议读不出 X。

---

## 6. 方法论局限 / 警告

> 本实验设计本身有几处妥协，结论应在这些上下文里解读。

1. **phyworld uniform_motion 是评估集，不是训练集**。1152 traj × 32 帧 = 36864 帧，相比 pusht 的 2.3M 帧少 **63 倍**。模型容量远超数据量，从 loss 曲线看也没有显著过拟合（因为 traj 内冗余高），但用更大数据（如 phyworld 的 30k–3M 视频生成集）会更可信。
2. **action 是合成的**。phyworld 物理上没有 action，velocity 是从位置反推出来的。这导致"模型学的是恒等式"而非"控制信号 → 状态变化"，预测任务高度退化。
3. **Probe 是线性的**。Ridge 回归只能测**线性可解码性**。如果 encoder 把信息编成非线性形式（如球位置的傅里叶特征），线性 probe 看不见，会低估 encoder 表征能力。改用 MLP probe 是常见的下一步。
4. **目标量 y/vy 是常量**：本质上 phyworld uniform_motion 是一个 1D（仅 x 轴）的运动数据集，2D probe target 里有一半维度是 trivial 的，让综合 R² 难以解读。如果换 2D 物理（碰撞、抛物线），所有维度都有信号。
5. **没和 phyworld 论文的方法对比**。论文用的是大型视频生成模型（CogVideoX 等）+ 不同协议（rollout 视频 + 物理量探针）。本实验只是"试试看 le-wm 怎么样"，不是 head-to-head 比较。

---

## 7. 后续实验建议

按"工程量 / 信息量"排序：

### 方案 A：换 phyworld 子任务（中等工程量，**强推荐**）

用 phyworld 的 **collision 或 parabola** 数据替换 uniform_motion。两个理由：
- 物理变非平凡：碰撞要学动量守恒，抛物线要学重力——velocity 不再是 trivial action
- 自带 ID/OOD 切分（论文核心评估），可直接对接
- 只需要：跑 [phyworld/id_ood_data/two_balls_collision.py](../phyworld/id_ood_data/two_balls_collision.py) 生成数据 → 改 [convert_to_lewm.py](../phyworld/scripts/convert_to_lewm.py) 适配新 schema → 改一个 yaml 配置

### 方案 B：去掉 action conditioning（小工程量，理论上更"纯"）

把 [le-wm/module.py 的 ARPredictor](../le-wm/module.py) 里 `block_class=ConditionalBlock` 换成 `Block`，让 predictor 不能依赖 action 作弊，被迫从 encoder emb 里学"运动信息"。
- 改 ~20 行代码
- 预计 vx R² 会显著上升（encoder 被迫编码速度）
- 但偏离了 le-wm 原始架构，结论的可推广性变差

### 方案 C：非线性 probe（小工程量，便宜的 sanity check）

把 probe 从 Ridge 换成 2-layer MLP，看 encoder 的 vx R² 会不会显著上升。
- 如果会：信息其实在 emb 里，只是非线性编码——可能就解释了为什么 le-wm 在 pusht 等任务上能 plan
- 如果不会：进一步确认 SIGReg 真的把速度信息扔掉了

---

## 8. 复现指南

### 环境
见 [wm/README.md](../README.md) 的 "Environment" 章节。venv 在 [le-wm/.venv](../le-wm/.venv)。

### 数据准备
```bash
cd /home/qlib/agent_memory/wm/phyworld
python scripts/convert_to_lewm.py
# → ~/.stable_worldmodel/phyworld_uniform_motion.h5 (100 MB)
```

### 训练
```bash
cd /home/qlib/agent_memory/wm/le-wm && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=<freest_gpu> WANDB_MODE=disabled \
  python -u train.py data=phyworld output_model_name=lewm_phyworld \
    subdir=phyworld_probe wandb.enabled=False trainer.max_epochs=20
```

### Probe
```bash
CUDA_VISIBLE_DEVICES=<freest_gpu> \
  python /home/qlib/agent_memory/wm/phyworld/scripts/probe_lewm_encoder.py \
    --ckpt ~/.stable_worldmodel/phyworld_probe/lewm_phyworld_epoch_20_object.ckpt
```

### 关键 artifacts
- 训练 ckpts: [~/.stable_worldmodel/phyworld_probe/lewm_phyworld_epoch_*_object.ckpt](~/.stable_worldmodel/phyworld_probe/) (1.4 GB total)
- 训练日志: [/tmp/lewm_phyworld_train.log](/tmp/lewm_phyworld_train.log)
- Probe 输出: [/tmp/probe_full.log](/tmp/probe_full.log)
- 转换脚本: [phyworld/scripts/convert_to_lewm.py](../phyworld/scripts/convert_to_lewm.py)
- Probe 脚本: [phyworld/scripts/probe_lewm_encoder.py](../phyworld/scripts/probe_lewm_encoder.py)
- 数据配置: [le-wm/config/train/data/phyworld.yaml](../le-wm/config/train/data/phyworld.yaml)

### 已知坑
1. **GPU 切换 OOM**：训练时根据 `nvidia-smi` 选 free memory ≥ 16 GB 的卡，不然会 OOM。
2. **checkpoint 自动 resume**：spt.Manager 会自动从 `<run_dir>/<output_model_name>_weights.ckpt` resume。换数据集时务必换 `output_model_name`，否则 action_dim 不匹配崩溃。
3. **Random encoder 走错分支**：probe 脚本里 `hasattr(model, 'encoder')` 对 ViTModel 也是 True（指向内层 ViTEncoder stack），需要显式 `hasattr(model, 'embeddings')` 判断（已修复）。

---

## 9. 致谢 / 参考

- LeWorldModel: Maes & Le Lidec et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* (2026 preprint). https://github.com/lucas-maes/le-wm
- PhyWorld: Kang & Yue et al., *How Far is Video Generation from World Model: A Physical Law Perspective* (arXiv:2411.02385). https://github.com/PhyWorld/PhyWorld
- SIGReg: Sketch Isotropic Gaussian Regularizer，le-wm 的核心创新，单 GPU 适用。
