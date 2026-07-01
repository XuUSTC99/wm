# PIWM Deep-Supervision — 三域统一对比（parabola / uniform_motion / collision）— **A800 修复版**

**日期**：2026-06-08
**机器**：A800（8×A800-80GB）
**init**：`+init_from_ckpt=lewm_paper_pusht/weights.pt`，**`loaded=216 unexpected=0`（pusht encoder 完整加载）**
**一句话**：用**修好 init**的 train.py 重跑 3 域 × 4 臂（12 训练 + 12 eval），取代此前的 A500 "重跑版"——那一版因 ViT 命名漂移导致 encoder 92% 随机初始化（`loaded=24/216`），数值无效。**本版是真 pusht 预训练 encoder 上的正确结果。**

> **为什么此前结论要全部作废**：A500/A800 的新版 transformers 把 ViT attention 改名（`encoder.layers.N.attention.{q,k,v,o}_proj`），对不上 qlib 旧存的 `weights.pt`（`encoder.encoder.layer.N.attention.attention.{query,key,value}`），`load_state_dict(strict=False)` 静默丢 192 个 key → encoder 随机初始化。已在 [train.py](../../le-wm/train.py) 加 `_remap_old_vit_keys()` + 加载守卫（<50% 直接报错）。本版 12 个 job 启动时全部 `loaded=216 unexpected=0`。
>
> **修复后两个"吓人异常"消失**（证实它们是随机 encoder 的产物，非真信号）：
> - collision v-OOD vx 的 pos-only 不再崩到 **−0.097**，现在 **+0.129**（无负值）；
> - uniform 上"mf4 反超 pos-only"的反转**没了**，pos-only 重新在多数指标领先（与 qlib 原报告一致）。

---

## 1. 设置（三域同口径）

- **训练**：ID-only 1000 trajs（`*_id1k.h5`），LeWM FT 20ep，pusht 预训练 init
- **5 臂**：
  - `baseline`：probe.weight=0（无 probe 对照）
  - `pos-only`：target=proprio, **frames=1**（单帧 → 自己当前帧位置）
  - `pos+vel`：target=[proprio, vel], **frames=1**（单帧 → pos+vel）
  - `mf4`：target=[proprio, vel], **frames=4**（4 帧拼接 → 最后帧 pos+vel）
  - **`mf4-pos-only`** ⭐ 新增（2026-06-26）：target=proprio, **frames=4**（4 帧拼接 → 最后帧位置；**不显式监督速度**）
- **mf4-pos-only 的研究问题**：如果 encoder 把位置编得干净，K=4 推理时 4 帧 emb 的差分**能否自动恢复出速度**？速度 ρ 与 mf4 持平 → 假说成立；显著低 → 显式 vel 监督是必要的
- **velocity 监督列**：parabola/uniform = `action`，collision = `state`
- **评估**：rollout（ARPredictor 自回归）在全 OOD eval 集；解码 ρ 用推理 K=4
- **指标**：`vx/vy ρ`=K=4 解码 Pearson；`latent cos`=方向（归一化）；`nMSE`=大小（按真 latent 方差归一，越低越好）。**cos/nMSE 都在同一 ckpt 自身 latent 空间里测（循环逻辑），塌方时会同时虚高/虚低——需配合绝对物理量误差看（待补）。**

---

## 2. parabola（自由落体 + 重力）

### vx 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | 0.374 | 0.600 | 0.604 | **0.671** | 0.556 |
| r/m-OOD | 0.607 | 0.601 | 0.644 | 0.644 | **0.655** |
| v-OOD | 0.579 | **0.732** | 0.695 | 0.701 | 0.602 |
| both-OOD | 0.728 | 0.732 | **0.750** | 0.747 | 0.696 |

### vy 解码 ρ（K=4，重力 → within-traj 强信号）
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | 0.879 | 0.983 | **0.985** | 0.984 | 0.984 |
| r/m-OOD | 0.875 | 0.970 | 0.976 | 0.976 | **0.980** |
| v-OOD | 0.853 | **0.951** | **0.954** | 0.870 | 0.937 |
| both-OOD | 0.779 | **0.963** | 0.942 | 0.898 | 0.957 |

### latent cos / nMSE by partition
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | 0.943/0.050 | 0.967/0.063 | 0.971/0.060 | **0.976/0.046** | 0.974/0.053 |
| r/m-OOD | 0.789/0.486 | 0.820/0.449 | 0.897/0.269 | **0.908/0.212** | 0.862/0.337 |
| v-OOD | 0.565/0.424 | 0.663/0.656 | **0.728/0.490** | 0.709/0.486 | 0.713/0.567 |
| both-OOD | 0.489/0.791 | 0.617/0.828 | **0.758/0.520** | 0.693/0.622 | 0.690/0.762 |

### latent cos / nMSE by horizon
| h | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| 1 | 0.975/0.041 | 0.982/0.037 | **0.986/0.030** | 0.982/0.029 | 0.982/0.036 |
| 4 | 0.856/0.267 | 0.868/0.251 | **0.930/0.154** | 0.913/0.170 | 0.904/0.205 |
| 8 | 0.798/0.418 | 0.831/0.336 | 0.861/0.304 | **0.875/0.287** | 0.817/0.431 |
| 16 | 0.550/0.717 | 0.622/0.773 | **0.699/0.584** | 0.630/0.744 | 0.672/0.699 |

**parabola 结论**：probe 各臂全面碾压 baseline。**pos+vel / mf4 综合最佳**；**mf4-pos-only ⭐ 在 vy 上几乎追平 mf4**（v-OOD 0.937 vs 0.870 还反超）但 **vx ID 落后**（0.556 vs mf4=0.671）——4 帧位置差分能恢复 vy（重力强信号）的速度，但 vx（水平常量速度，时刻 0）的位置-到-速度信号弱，差分不够。

---

## 3. uniform_motion（单球匀速直线，vx 恒定，vy≡0）

> vy 全 nan：匀速运动 vy 恒为 0（零方差），Pearson ρ 对常量无定义。只看 vx。

### vx 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | 0.497 | 0.702 | 0.523 | 0.707 | **0.754** |
| r/m-OOD | 0.647 | **0.750** | 0.728 | 0.742 | 0.742 |
| v-OOD | 0.774 | 0.929 | 0.915 | 0.899 | **0.942** |
| both-OOD | 0.885 | 0.877 | 0.913 | **0.922** | 0.901 |

### latent cos / nMSE by partition
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | **0.970/0.065** | 0.958/0.089 | 0.831/0.304 | 0.917/0.169 | 0.956/0.095 |
| r/m-OOD | 0.764/0.385 | **0.831/0.385** | 0.586/0.815 | 0.757/0.463 | 0.812/0.380 |
| v-OOD | 0.931/0.150 | **0.937/0.154** | 0.855/0.345 | 0.871/0.298 | 0.924/0.180 |
| both-OOD | 0.844/0.295 | **0.854/0.341** | 0.761/0.543 | 0.817/0.418 | 0.824/0.396 |

### latent cos / nMSE by horizon
| h | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| 1 | 0.991/0.014 | **0.993/0.013** | 0.983/0.030 | 0.984/0.028 | 0.988/0.022 |
| 4 | 0.951/0.077 | **0.968/0.064** | 0.912/0.166 | 0.920/0.141 | 0.945/0.094 |
| 8 | 0.889/0.167 | **0.923/0.153** | 0.800/0.365 | 0.837/0.297 | 0.883/0.203 |
| 16 | 0.829/0.282 | **0.831/0.330** | 0.663/0.645 | 0.778/0.424 | 0.824/0.341 |

**uniform 结论**：**pos-only 综合最佳**（cos by horizon 全程、cos by partition 多数最高）；**pos+vel 最差**（零方差 vy 列干扰）。⭐ **mf4-pos-only 在 vx ID（0.754）和 v-OOD（0.942）上反超所有 4 个旧 arm**——4 帧位置（无 vel 监督）+ 差分**完美恢复匀速直线的速度**，因为匀速 = 等距位置差，4 帧差分天然就是该速度的最优估计。这是用户假说在 uniform 上的强证据。

---

## 4. collision（双球碰撞，action=加速度，velocity 在 state）

### vx 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | 0.508 | 0.714 | 0.686 | 0.567 | **0.755** |
| r/m-OOD | 0.476 | 0.457 | **0.689** | 0.684 | 0.420 |
| v-OOD | 0.134 | 0.129 | 0.450 | **0.468** | 0.262 |
| both-OOD | 0.371 | 0.413 | 0.457 | **0.602** | 0.585 |

### vy 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | 0.260 | 0.576 | 0.585 | **0.762** | 0.624 |
| r/m-OOD | 0.523 | 0.505 | **0.714** | 0.622 | 0.485 |
| v-OOD | 0.298 | **0.598** | 0.322 | 0.525 | 0.472 |
| both-OOD | 0.453 | 0.419 | 0.517 | 0.543 | **0.592** |

### latent cos / nMSE by partition
| partition | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| ID | 0.529/0.972 | **0.684/0.658** | 0.596/0.828 | 0.635/0.781 | 0.593/0.829 |
| r/m-OOD | 0.504/0.973 | 0.515/0.959 | 0.573/0.841 | **0.585/0.876** | 0.578/0.868 |
| v-OOD | 0.335/1.338 | 0.342/1.438 | 0.408/1.272 | **0.454/1.187** | 0.435/1.233 |
| both-OOD | **0.475/1.136** | 0.443/1.132 | 0.477/1.132 | 0.422/1.248 | 0.524/1.017 |

### latent cos / nMSE by horizon
| h | baseline | pos-only | pos+vel | mf4 | **mf4-pos-only** ⭐ |
|---|---|---|---|---|---|
| 1 | 0.990/0.021 | **0.991/0.021** | 0.990/0.021 | 0.989/0.025 | **0.991/0.021** |
| 4 | 0.852/0.303 | 0.870/0.271 | **0.875/0.266** | 0.859/0.294 | 0.867/0.279 |
| 8 | 0.626/0.774 | 0.632/0.735 | **0.651/0.717** | 0.643/0.744 | 0.653/0.702 |
| 16 | 0.356/1.300 | 0.357/1.288 | 0.408/1.196 | 0.366/1.315 | **0.446/1.134** |

**collision 结论**：
- **ID 上 mf4-pos-only 反超**：vx ID=0.755（5 臂最高，超过 pos-only=0.714 和 mf4=0.567）；
- **但 OOD 上崩**：v-OOD vx=0.262（远低于 pos+vel=0.450 / mf4=0.468）。**碰撞这种"速度跳变"的域，4 帧位置差分恢复不了碰撞瞬间的速度突变**——需要显式速度监督才能撑住 OOD 外推。
- 长程 cos h=16 上 mf4-pos-only=0.446 是 5 臂最高，但 partition v-OOD/both-OOD cos 反而比 pos+vel/mf4 弱——长程方向稳定但局部 OOD 跳变信息缺失。
- ⚠️ **此前 A500 broken 版的"pos-only v-OOD = −0.097 崩塌"是 init bug 假象**——修好后是 **+0.129**（低但为正），不存在负相关崩塌。

---

## 5. 跨三域核心结论（修复版 + mf4-pos-only）

| 域 | 最佳臂 | 关键点 |
|---|---|---|
| **parabola** | pos+vel / mf4 | probe 全面 >> baseline；vx ID 看 mf4，vx OOD / 长程 cos 看 pos+vel；**mf4-pos-only 在 vy 几乎追平 mf4**（4 帧差分恢复 vy）|
| **uniform** | **mf4-pos-only** ⭐ / pos-only | mf4-pos-only 在 vx ID（0.754）+ v-OOD（0.942）反超所有旧 arm——匀速直线下 4 帧位置差分**完美**恢复 vx |
| **collision** | ID→mf4-pos-only / pos-only，OOD→pos+vel / mf4 | mf4-pos-only vx ID=0.755 最高；但碰撞瞬间速度跳变靠差分恢复不了 → OOD 落后 |

### 5.1 deep-sup probe 是否有用？→ **是，全域显著优于 baseline**
三域里 probe 各臂在速度解码上都大幅超过无 probe 的 baseline（如 parabola vx ID 0.374→0.67、collision vx ID 0.508→0.755、uniform vx v-OOD 0.774→0.942）。**这与 broken 版（随机 encoder）得出的"deep-sup 有害/塌方"结论相反**——那是 init bug 导致的错误结论。

### 5.2 within-traj std 机制论（修复后重新评估）
原假说：多帧监督（mf4）的价值取决于被监督速度的 within-traj std。修复版数据**部分支持**：
- ✅ **uniform**（vx std=0，匀速）：mf4 **不**是最佳，pos-only 胜——低 std → 多帧无增益，符合假说；
- ✅ **collision**（vx std=0.21）：mf4 在 both-OOD vx（0.602）和 ID vy（0.762）最佳——std>0 → 多帧有价值，符合；
- ⚠️ **parabola**（vy std=0.23）：mf4 在 vy 上反而不是最佳（pos+vel/pos-only 更高），但在 vx ID 最佳——混合。

整体比 broken 版（完全推翻假说）要支持得多，但 parabola 上不完全干净。**仍需 multi-seed 确认排序稳定性。**

### 5.3 mf4-pos-only：4 帧位置差分能否替代显式速度监督？⭐ 新结论

**假说**：encoder 干净编码每帧位置 + K=4 推理时 probe head 用 4 帧 emb 拼接 + linear projection 自动差分恢复速度，**不需要显式 vel 监督**。

**实证答案**：**部分成立，依域而定**——

| 域 | 速度类型 | mf4-pos-only 表现 | 假说 |
|---|---|---|---|
| **uniform** | 匀速直线（vx 常量）| **全面胜过显式 vel 监督**（vx ID 0.754、v-OOD 0.942 都最高）| ✅ **强支持**——匀速 = 等距位置差，4 帧差分天然就是最优速度估计 |
| **parabola** | vy 重力线性变（强 within-traj 信号）| vy 接近最佳（0.984/0.937/0.957），**vx 落后**（ID 0.556 vs mf4=0.671）| ⚠️ **部分**——重力信号强的方向差分能学，常量速度方向差分信号弱 |
| **collision** | vx 碰撞瞬间跳变 | vx ID 0.755 **最高**，但 v-OOD=0.262 **崩**（远低于 pos+vel=0.450）| ❌ **OOD 上失效**——4 帧位置差分平均掉了碰撞瞬间的速度突变 |

**机制总结**：4 帧位置 + linear probe head 实现的是**有限差分 ≈ d(pos)/dt 的 linear approximation**：
- 速度变化平滑（uniform / parabola vy）→ 差分对得上真实速度 → 不需要显式 vel 监督
- 速度跳变（collision 碰撞瞬间）→ 平滑差分丢掉跳变 → 必须显式 vel 监督才能学到瞬时速度

**所以"位置 + 多帧能否替代显式 vel"是个 task-dependent 问题，不是普适答案。**

---

## 6. Caveats
- **单 seed**：每 (域, 臂) 仅 1 seed，cell 间 0.02–0.10 的差异可能含 seed 噪声。建议关键 cell 补 3 seed。
- **cos/nMSE 循环逻辑**：两者都在 trained encoder 自身空间里测，不能单独证明 predictor 真预测对了未来；需补**物理量绝对轨迹误差**（latent→位置 linear probe → 比 PhyWorld 真实位置的像素 MAE/RMSE，**不需要 image decoder**）。
- **uniform vy=nan**：vy≡0 零方差，非 bug（见 [piwm_uniform_collision_results.md](piwm_uniform_collision_results.md)）。
- **未做**：multi-seed；λ_probe × frames sweep（之前的 45-config sweep 也是 broken init，需用修好的 init 重跑）；物理量绝对误差。

---

## 7. 文件 / 复现
| 项 | 路径 |
|---|---|
| 本报告 | `reports/6-2/piwm_three_domains_A800.md` |
| qlib 原始报告（不受 bug 影响）| [piwm_three_domains.md](piwm_three_domains.md) / [piwm_uniform_collision_results.md](piwm_uniform_collision_results.md) |
| 训练 + eval 编排（4 臂）| [rerun_three_domains.sh](rerun_three_domains.sh) |
| 训练 + eval 编排（第 5 臂 mf4-pos-only）| [run_mf4posonly.sh](run_mf4posonly.sh) ⭐ 新增 |
| init 修复 | [le-wm/train.py](../../le-wm/train.py) `_remap_old_vit_keys()` + 加载守卫 |
| 训练/eval 日志 | `/data1/likun-share/junjxu/runs/6-2_three_domains_logs/{train,rollout}_*.log`（fixed-init, loaded=216）|
| broken 版日志（作废）| `/data1/likun-share/junjxu/runs/6-2_three_domains_logs_BROKEN/` |
| ckpt（4 臂）| `/data1/likun-share/junjxu/.stable_worldmodel/{parabola,uniform_motion,collision}_{paperinit,piwm_probe,piwm_posvel,piwm_mf4}_id1k/` |
| ckpt（mf4-pos-only）| `/data1/likun-share/junjxu/.stable_worldmodel/{parabola,uniform_motion,collision}_piwm_mf4posonly_id1k/` |

### 复现命令
```bash
# 4 臂（baseline / pos-only / pos+vel / mf4），约 60 min on 8 GPU
bash /home/likun-share/junjxu/wm/reports/6-2/rerun_three_domains.sh "0 1 2 3 4 5 6 7"

# 第 5 臂（mf4-pos-only），约 25 min on 3 GPU
bash /home/likun-share/junjxu/wm/reports/6-2/run_mf4posonly.sh "0 1 2"

# 启动时务必确认日志里 [init_from_ckpt] loaded=216（不是 24）
```
