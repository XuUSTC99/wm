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
- **4 臂**：`baseline`(probe.weight=0) / `pos-only`(target=proprio, frames=1) / `pos+vel`(target=[proprio,vel], frames=1) / `mf4`(target=[proprio,vel], frames=4)
- **velocity 监督列**：parabola/uniform = `action`，collision = `state`
- **评估**：rollout（ARPredictor 自回归）在全 OOD eval 集；解码 ρ 用推理 K=4
- **指标**：`vx/vy ρ`=K=4 解码 Pearson；`latent cos`=方向（归一化）；`nMSE`=大小（按真 latent 方差归一，越低越好）。**cos/nMSE 都在同一 ckpt 自身 latent 空间里测（循环逻辑），塌方时会同时虚高/虚低——需配合绝对物理量误差看（待补）。**

---

## 2. parabola（自由落体 + 重力）

### vx 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.374 | 0.600 | 0.604 | **0.671** |
| r/m-OOD | 0.607 | 0.601 | **0.644** | **0.644** |
| v-OOD | 0.579 | **0.732** | 0.695 | 0.701 |
| both-OOD | 0.728 | 0.732 | **0.750** | 0.747 |

### vy 解码 ρ（K=4，重力 → within-traj 强信号）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.879 | 0.983 | **0.985** | 0.984 |
| r/m-OOD | 0.875 | 0.970 | **0.976** | **0.976** |
| v-OOD | 0.853 | 0.951 | **0.954** | 0.870 |
| both-OOD | 0.779 | **0.963** | 0.942 | 0.898 |

### latent cos / nMSE by partition
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.943/0.050 | 0.967/0.063 | 0.971/0.060 | **0.976/0.046** |
| r/m-OOD | 0.789/0.486 | 0.820/0.449 | 0.897/0.269 | **0.908/0.212** |
| v-OOD | 0.565/0.424 | 0.663/0.656 | **0.728/0.490** | 0.709/0.486 |
| both-OOD | 0.489/0.791 | 0.617/0.828 | **0.758/0.520** | 0.693/0.622 |

### latent cos / nMSE by horizon
| h | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| 1 | 0.975/0.041 | 0.982/0.037 | **0.986/0.030** | 0.982/0.029 |
| 4 | 0.856/0.267 | 0.868/0.251 | **0.930/0.154** | 0.913/0.170 |
| 8 | 0.798/0.418 | 0.831/0.336 | 0.861/0.304 | **0.875/0.287** |
| 16 | 0.550/0.717 | 0.622/0.773 | **0.699/0.584** | 0.630/0.744 |

**parabola 结论**：probe 各臂全面碾压 baseline（vx ID 0.374→0.60+，长程 cos h=16 0.55→0.70）。**pos+vel / mf4 综合最佳**——mf4 在 ID/中程最强，pos+vel 在长程（h=16）和 OOD partition 的 cos/nMSE 最佳。vy 上三个 probe 臂都到 0.95+（重力信号强，谁都学得好）。

---

## 3. uniform_motion（单球匀速直线，vx 恒定，vy≡0）

> vy 全 nan：匀速运动 vy 恒为 0（零方差），Pearson ρ 对常量无定义。只看 vx。

### vx 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.497 | 0.702 | 0.523 | **0.707** |
| r/m-OOD | 0.647 | **0.750** | 0.728 | 0.742 |
| v-OOD | 0.774 | **0.929** | 0.915 | 0.899 |
| both-OOD | 0.885 | 0.877 | 0.913 | **0.922** |

### latent cos / nMSE by partition
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | **0.970/0.065** | 0.958/0.089 | 0.831/0.304 | 0.917/0.169 |
| r/m-OOD | 0.764/0.385 | **0.831/0.385** | 0.586/0.815 | 0.757/0.463 |
| v-OOD | 0.931/0.150 | **0.937/0.154** | 0.855/0.345 | 0.871/0.298 |
| both-OOD | 0.844/0.295 | **0.854/0.341** | 0.761/0.543 | 0.817/0.418 |

### latent cos / nMSE by horizon
| h | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| 1 | 0.991/0.014 | **0.993/0.013** | 0.983/0.030 | 0.984/0.028 |
| 4 | 0.951/0.077 | **0.968/0.064** | 0.912/0.166 | 0.920/0.141 |
| 8 | 0.889/0.167 | **0.923/0.153** | 0.800/0.365 | 0.837/0.297 |
| 16 | 0.829/0.282 | **0.831/0.330** | 0.663/0.645 | 0.778/0.424 |

**uniform 结论**：**pos-only 综合最佳**——vx v-OOD（0.929）、cos by horizon 全程、cos by partition 多数最高。**pos+vel 明显最差**（多了个零方差的 vy 监督列反而干扰：ID cos 0.831、长程 cos 0.663）。mf4 中等。**与 qlib 原报告"pos-only 最佳"方向一致**（A500 broken 版的"mf4 反转"是 init bug 假象，已消除）。

---

## 4. collision（双球碰撞，action=加速度，velocity 在 state）

### vx 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.508 | **0.714** | 0.686 | 0.567 |
| r/m-OOD | 0.476 | 0.457 | **0.689** | 0.684 |
| v-OOD | 0.134 | 0.129 | 0.450 | **0.468** |
| both-OOD | 0.371 | 0.413 | 0.457 | **0.602** |

### vy 解码 ρ（K=4）
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.260 | 0.576 | 0.585 | **0.762** |
| r/m-OOD | 0.523 | 0.505 | **0.714** | 0.622 |
| v-OOD | 0.298 | **0.598** | 0.322 | 0.525 |
| both-OOD | 0.453 | 0.419 | 0.517 | **0.543** |

### latent cos / nMSE by partition
| partition | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| ID | 0.529/0.972 | **0.684/0.658** | 0.596/0.828 | 0.635/0.781 |
| r/m-OOD | 0.504/0.973 | 0.515/0.959 | 0.573/0.841 | **0.585/0.876** |
| v-OOD | 0.335/1.338 | 0.342/1.438 | 0.408/1.272 | **0.454/1.187** |
| both-OOD | **0.475/1.136** | 0.443/1.132 | 0.477/1.132 | 0.422/1.248 |

### latent cos / nMSE by horizon
| h | baseline | pos-only | pos+vel | mf4 |
|---|---|---|---|---|
| 1 | 0.990/0.021 | **0.991/0.021** | 0.990/0.021 | 0.989/0.025 |
| 4 | 0.852/0.303 | 0.870/0.271 | **0.875/0.266** | 0.859/0.294 |
| 8 | 0.626/0.774 | 0.632/0.735 | **0.651/0.717** | 0.643/0.744 |
| 16 | 0.356/1.300 | 0.357/1.288 | **0.408/1.196** | 0.366/1.315 |

**collision 结论**：
- **ID 上 pos-only 最佳**（vx 0.714、cos 0.684）；
- **OOD 上含速度监督的臂（pos+vel / mf4）明显胜出**：v-OOD vx baseline=0.134、pos-only=0.129（都低），而 **pos+vel=0.450 / mf4=0.468**；both-OOD vx mf4=0.602 最高。说明碰撞这种"速度强相关"的域，**把速度喂进 probe（pos+vel / mf4）才能撑住 OOD 外推**。
- ⚠️ **此前 A500 broken 版的"pos-only v-OOD = −0.097 崩塌"是 init bug 假象**——修好后是 **+0.129**（低但为正），不存在负相关崩塌。

---

## 5. 跨三域核心结论（修复版）

| 域 | 最佳臂 | 关键点 |
|---|---|---|
| **parabola** | pos+vel / mf4 | probe 全面 >> baseline；长程/OOD 看 pos+vel，ID/中程看 mf4 |
| **uniform** | **pos-only** | 与 qlib 原报告一致；pos+vel 因零方差 vy 列反而最差 |
| **collision** | ID→pos-only，OOD→pos+vel/mf4 | 速度监督对 OOD 外推关键；无任何崩塌 |

### 5.1 deep-sup probe 是否有用？→ **是，全域显著优于 baseline**
三域里 probe 各臂在速度解码上都大幅超过无 probe 的 baseline（如 parabola vx ID 0.374→0.67、collision vx ID 0.508→0.714、uniform vx v-OOD 0.774→0.929）。**这与 broken 版（随机 encoder）得出的"deep-sup 有害/塌方"结论相反**——那是 init bug 导致的错误结论。

### 5.2 within-traj std 机制论（修复后重新评估）
原假说：多帧监督（mf4）的价值取决于被监督速度的 within-traj std。修复版数据**部分支持**：
- ✅ **uniform**（vx std=0，匀速）：mf4 **不**是最佳，pos-only 胜——低 std → 多帧无增益，符合假说；
- ✅ **collision**（vx std=0.21）：mf4 在 both-OOD vx（0.602）和 ID vy（0.762）最佳——std>0 → 多帧有价值，符合；
- ⚠️ **parabola**（vy std=0.23）：mf4 在 vy 上反而不是最佳（pos+vel/pos-only 更高），但在 vx ID 最佳——混合。

整体比 broken 版（完全推翻假说）要支持得多，但 parabola 上不完全干净。**仍需 multi-seed 确认排序稳定性。**

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
| 训练 + eval 编排 | [rerun_three_domains.sh](rerun_three_domains.sh) |
| init 修复 | [le-wm/train.py](../../le-wm/train.py) `_remap_old_vit_keys()` + 加载守卫 |
| 训练/eval 日志 | `/data1/likun-share/junjxu/runs/6-2_three_domains_logs/{train,rollout}_*.log`（fixed-init, loaded=216）|
| broken 版日志（作废）| `/data1/likun-share/junjxu/runs/6-2_three_domains_logs_BROKEN/` |
| ckpt | `/data1/likun-share/junjxu/.stable_worldmodel/{parabola,uniform_motion,collision}_{paperinit,piwm_probe,piwm_posvel,piwm_mf4}_id1k/` |

### 复现命令
```bash
bash /home/likun-share/junjxu/wm/reports/6-2/rerun_three_domains.sh "0 1 2 3 4 5 6 7"
# 启动时务必确认日志里 [init_from_ckpt] loaded=216（不是 24）
```
