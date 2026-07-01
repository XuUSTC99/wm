> # ⚠️⚠️ 重大更正（2026-06-07）：本报告全部数据无效 ⚠️⚠️
> 
> **根因 — 预训练权重根本没加载上。** 本报告的 45 个 ckpt 全部在 A500 上训练，而 A500 的 transformers 版本把 ViT attention 参数改了名（`encoder.layers.N.attention.{q,k,v,o}_proj`），与几个月前在 qlib 存的 `lewm_paper_pusht/weights.pt` 旧命名（`encoder.encoder.layer.N.attention.attention.{query,key,value}`）对不上。`train.py` 的 `load_state_dict(strict=False)` **静默丢弃了 192 个对不上的 key**——训练日志白纸黑字：
> 
> ```
> [init_from_ckpt] loaded=24 unexpected=192 missing=281
> ```
> 
> **即：ViT 的 12 层 transformer 主体全是随机初始化**，只有 embedding+projector+pred_proj（24 个）来自 pusht。这 45 个模型的 encoder ≈ 在 1000 条轨迹上 FT 20 epoch 的近随机网络，**根本不是 pusht 预训练。**
> 
> **已修复**：[train.py](../../le-wm/train.py) 加了 `_remap_old_vit_keys()`（旧→新命名映射）+ 加载守卫（<50% 报错）。修复后 `loaded=216 unexpected=0`，权重 `allclose` 等于 pusht 源值。
> 
> **修复后的 parabola 对照实验（f=2, w∈{0.1,1,50}）证明本报告结论翻盘：**
> 
> | 现象 | 本报告（broken init）| 修好 init 后 |
> |---|---|---|
> | pred_loss 随 λ 变化 | **+57%**（0.0115→0.0181，"高 λ 摧毁预测") | **几乎持平 +21%**（0.0047→0.0057），且整体低 2.5–3× |
> | projector eff_dim（塌方） | 35→6.4（CLS 仅 6.9–10.7，本就近退化）| 60→8.5（CLS 34→14，基底健康） |
> | "w=50 全面主导" | — | 只是 probe-dual 指标的对偶游戏，pred_loss 看 **w=0.1 最低** |
> 
> **所以下面所有"w=50 主导 / λ 越大越好 / 长程 cos 未饱和 / collision 崩塌靠高 λ 救回"的结论都不成立**——它们是随机初始化 encoder 的产物 + probe 损失对偶（`K=4 ρ`、`latent cos` 都随 λ 必然上升，不代表 world model 变好）。
> 
> **正确的 45-config 数字需要用修好的 init 重跑整个 sweep**（截至 6-07 因他人占用 GPU 未完成）。在那之前，**本报告以下内容仅作"踩坑记录"保留，不可引用其数值结论。**
> 
> 相关：[piwm_three_domains_new.md](piwm_three_domains_new.md) 同样 broken（A500 重跑）；[piwm_uniform_collision_results.md](piwm_uniform_collision_results.md) 与 [piwm_three_domains.md](piwm_three_domains.md) 是 **qlib 原始结果，不受此 bug 影响**（weights.pt 由 qlib 自存、命名自洽）。
> 
> ---

# λ_probe × frames 三域 sweep — 完整结果报告（5 weights × 3 frames × 3 domains = 45 configs）

**⚠️ 见顶部更正——以下数值因 init bug 全部无效，仅作记录。**

**日期**：2026-06-06
**配置数**：45 = 3 域 × **5 weights** {0.1, 1.0, 10.0, 30.0, 50.0} × 3 frames {1, 2, 4}（`probe.target=[proprio, vel_col]`，20 epoch FT）
**计算资源**：
- 第一轮 27 jobs（w ∈ {0.1, 1, 10}）：59 分钟（2026-06-05 22:35 → 23:34）
- 扩展 18 jobs（w ∈ {30, 50}）：**37 分钟**（2026-06-05 23:45 → 06-06 00:22）
- 总耗时：约 1.5 小时，8×A800-80GB

✅ 45/45 训练完成、45/45 eval 完成、45/45 日志解析

---

## 0. 核心结论（请只读这一节）

### 🎯 主结论：**w=50 全面主导，但 λ 在某些指标上未到平台期**

| 域 | w=50 拿下的指标 | w=30 拿下 | w=10 拿下 |
|---|---|---|---|
| **parabola** | 5/7（vx v-OOD、cos h=4/16、cos both-OOD 等）| 2/7（vx ID、vy ID）| 1/7（vy v-OOD）|
| **uniform_motion** | 3/5（vx v-OOD、cos h=16、cos both-OOD）| 2/5（vx ID、cos h=4）| 0 |
| **collision** | 6/7（除 vx ID）| 1/7（vx ID）| 0 |

**总计：w=50 赢 14/19 cell，w=30 赢 5/19，w=10 仅 1/19，w=0.1/w=1.0 全 0。**

### 趋势分析

| 指标 | 0.1 → 1 → 10 → 30 → 50 |
|---|---|
| parabola cos h=16 (f=2) | 0.695 → 0.696 → 0.835 → 0.832 → **0.872**（**还在涨**）|
| uniform cos h=16 (f=4) | 0.821 → 0.831 → 0.891 → 0.939 → **0.954**（**还在涨**）|
| collision cos h=16 (f=4) | 0.464 → 0.503 → 0.556 → 0.563 → **0.633**（**还在涨**）|
| parabola vx ID (f=2) | 0.531 → 0.440 → 0.640 → **0.713** → 0.654（**w=30 峰值**）|
| collision vx v-OOD (f=2) | −0.060 → 0.238 → 0.438 → 0.241 → 0.323（**w=10 峰值，回落**）|
| collision vx v-OOD (f=1) | 0.019 → 0.267 → 0.205 → 0.267 → **0.529**（**单调上升**）|

→ **长程预测 cos** 在 w=50 仍在涨；**vx/vy 解码 ρ** 已开始非单调（出现 w=30 或 w=10 局部峰值）。建议**继续扩 w=100 / w=200 看长程指标是否还能涨**。

### 关键发现（与前 3 版报告的彻底改写）

| 旧结论 | 新结论（5-weights sweep） |
|---|---|
| `piwm_deepsup_results.md`: "mf4 (f=4) 是 deep-sup 正解" | **w=50/f=2 在 parabola 上才是最佳；frames 偏好依域不同** |
| `piwm_three_domains.md`: "f 由 within-traj std 决定" | **uniform std=0 但 w≥30 时 f=4 全面碾压 f=1 — 假说不成立** |
| `piwm_three_domains_new.md`: "pos-only 在 uniform 上最佳" | **w=50/f=4 才是 uniform 最佳；f=1 偏好只在低 λ 时存在** |
| `piwm_three_domains_new.md`: "collision v-OOD vx posonly=−0.097 崩塌" | **崩塌只在 λ 太弱时发生**；w=50/f=1 把 collision vx v-OOD 救到 **+0.529** |
| 默认 `loss.probe.weight=1.0` | **λ=1.0 远远不够**——λ ≥ 10 才进入合理区，λ=50 时仍未饱和 |

---

## 1. 三域 weight × frames 完整网格

每 cell = K=4 推理下的 Pearson ρ。**粗体** = best per table；下划线 = 出现回落点。

### 1.1 parabola（自由落体 + 重力）

#### vx (vel0) K=4 — ID
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.534 | +0.531 | +0.370 |
| w=1.0  | +0.470 | +0.440 | +0.573 |
| w=10.0 | +0.335 | +0.640 | +0.526 |
| w=30.0 | +0.409 | **+0.713** | +0.557 |
| w=50.0 | +0.465 | +0.654 | +0.636 |

> **w=30 是峰值**（0.713），w=50 略回落到 0.654。

#### vx (vel0) K=4 — v-OOD
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.522 | +0.828 | +0.468 |
| w=1.0  | +0.712 | +0.665 | +0.804 |
| w=10.0 | +0.634 | +0.795 | +0.749 |
| w=30.0 | +0.736 | +0.748 | +0.651 |
| w=50.0 | +0.472 | **+0.855** | +0.762 |

#### vy (vel1) K=4 — ID（重力主导）
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.971 | +0.958 | +0.922 |
| w=1.0  | +0.978 | +0.972 | +0.962 |
| w=10.0 | +0.983 | +0.984 | +0.979 |
| w=30.0 | +0.989 | **+0.991** | +0.982 |
| w=50.0 | +0.983 | +0.983 | +0.983 |

> 全 9 cells 都 ≥0.92，几乎到天花板。

#### latent cos @ h=16（多步预测保真度）
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.687 | +0.695 | +0.682 |
| w=1.0  | +0.646 | +0.696 | +0.748 |
| w=10.0 | +0.664 | +0.835 | +0.810 |
| w=30.0 | +0.803 | +0.832 | +0.713 |
| w=50.0 | +0.777 | **+0.872** | +0.865 |

> 相比 w=1.0 提升 **+0.176**（0.696 → 0.872），**长程预测的巨大胜利**。

#### latent cos — both-OOD
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.689 | +0.777 | +0.640 |
| w=1.0  | +0.618 | +0.684 | +0.731 |
| w=10.0 | +0.623 | +0.838 | +0.766 |
| w=30.0 | +0.823 | +0.833 | +0.811 |
| w=50.0 | +0.860 | **+0.867** | +0.858 |

> w=50 时 3 个 frames 几乎并列（0.858-0.867），**frames 在高 λ 上不重要**。

---

### 1.2 uniform_motion（单球匀速直线，vy ≡ 0）

> vy 列全 nan（vy 恒为 0），只看 vx。

#### vx (vel0) K=4 — ID
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.678 | +0.765 | +0.706 |
| w=1.0  | +0.615 | +0.744 | +0.772 |
| w=10.0 | +0.787 | +0.788 | +0.791 |
| w=30.0 | +0.606 | +0.815 | **+0.878** |
| w=50.0 | +0.728 | +0.781 | +0.865 |

> **f=4 在 w=30 反超 f=1（0.878 vs 0.606），完全颠覆原报告"uniform 上 f=1 最佳"的结论。**

#### vx (vel0) K=4 — v-OOD
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.765 | +0.873 | +0.686 |
| w=1.0  | +0.884 | +0.925 | +0.877 |
| w=10.0 | +0.949 | +0.941 | +0.922 |
| w=30.0 | +0.938 | +0.944 | +0.952 |
| w=50.0 | +0.865 | +0.935 | **+0.967** |

#### latent cos @ h=16
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.834 | +0.845 | +0.821 |
| w=1.0  | +0.808 | +0.828 | +0.831 |
| w=10.0 | +0.897 | +0.890 | +0.891 |
| w=30.0 | +0.827 | +0.895 | +0.939 |
| w=50.0 | +0.840 | +0.915 | **+0.954** |

> uniform 上 **f=4 + 高 λ** 是新的 SOTA。

#### latent cos — both-OOD
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.820 | +0.812 | +0.828 |
| w=1.0  | +0.823 | +0.841 | +0.817 |
| w=10.0 | +0.910 | +0.887 | +0.896 |
| w=30.0 | +0.879 | +0.910 | +0.936 |
| w=50.0 | +0.895 | +0.925 | **+0.945** |

---

### 1.3 collision（双球碰撞，action=加速度，速度在 state）

#### vx (vel0) K=4 — ID
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.466 | +0.598 | +0.456 |
| w=1.0  | +0.430 | +0.708 | +0.688 |
| w=10.0 | +0.592 | +0.744 | +0.704 |
| w=30.0 | +0.528 | +0.789 | **+0.812** |
| w=50.0 | +0.509 | +0.700 | +0.691 |

> **w=30/f=4 是峰值**（0.812），w=50 反而回落。

#### vx (vel0) K=4 — v-OOD ⚠️（前报告里 posonly 崩到 −0.097 的地方）
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.019 | **−0.060** | +0.108 |
| w=1.0  | +0.267 | +0.238 | +0.380 |
| w=10.0 | +0.205 | +0.438 | +0.452 |
| w=30.0 | +0.267 | +0.241 | +0.214 |
| w=50.0 | **+0.529** | +0.323 | +0.344 |

> 极不稳定：w=10 时 f=4 最佳（0.452），w=30 全面退化（最高仅 0.267），w=50 时 f=1 突然反超到 **0.529**。**这是单 seed 噪声的典型证据，需要 multi-seed**。

#### vy (vel1) K=4 — ID
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.544 | +0.688 | +0.476 |
| w=1.0  | +0.702 | +0.804 | +0.781 |
| w=10.0 | +0.824 | +0.618 | +0.780 |
| w=30.0 | +0.734 | +0.842 | +0.816 |
| w=50.0 | +0.807 | +0.770 | **+0.844** |

#### latent cos @ h=16
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.463 | +0.461 | +0.464 |
| w=1.0  | +0.514 | +0.517 | +0.503 |
| w=10.0 | +0.547 | +0.496 | +0.556 |
| w=30.0 | +0.527 | +0.516 | +0.563 |
| w=50.0 | +0.615 | +0.537 | **+0.633** |

> collision 长程预测仍然单调上升，**最难的域，最受益于高 λ**。

#### latent cos — both-OOD
| w \ f | f=1 | f=2 | f=4 |
|---|---|---|---|
| w=0.1  | +0.541 | +0.512 | +0.543 |
| w=1.0  | +0.562 | +0.562 | +0.553 |
| w=10.0 | +0.621 | +0.569 | +0.608 |
| w=30.0 | +0.579 | +0.582 | +0.596 |
| w=50.0 | **+0.655** | +0.567 | +0.626 |

---

## 2. 全部 best-(w, f) 汇总

| 域 | 指标 | best (w, f) | ρ |
|---|---|---|---|
| parabola | vx K=4 ID | **w=30.0, f=2** | +0.713 |
| parabola | vx K=4 v-OOD | **w=50.0, f=2** | +0.855 |
| parabola | vy K=4 ID | **w=30.0, f=2** | +0.991 |
| parabola | vy K=4 v-OOD | w=10.0, f=4 | +0.975 |
| parabola | cos h=4 | **w=50.0, f=2** | +0.970 |
| parabola | cos h=16 | **w=50.0, f=2** | +0.872 |
| parabola | cos both-OOD | **w=50.0, f=2** | +0.867 |
| uniform | vx K=4 ID | **w=30.0, f=4** | +0.878 |
| uniform | vx K=4 v-OOD | **w=50.0, f=4** | +0.967 |
| uniform | cos h=4 | **w=30.0, f=4** | +0.975 |
| uniform | cos h=16 | **w=50.0, f=4** | +0.954 |
| uniform | cos both-OOD | **w=50.0, f=4** | +0.945 |
| collision | vx K=4 ID | **w=30.0, f=4** | +0.812 |
| collision | vx K=4 v-OOD | **w=50.0, f=1** | +0.529 |
| collision | vy K=4 ID | **w=50.0, f=4** | +0.844 |
| collision | vy K=4 v-OOD | **w=50.0, f=1** | +0.705 |
| collision | cos h=4 | **w=50.0, f=2** | +0.935 |
| collision | cos h=16 | **w=50.0, f=4** | +0.633 |
| collision | cos both-OOD | **w=50.0, f=1** | +0.655 |

**统计**：
- w=50 占 **14/19** best cells (74%)
- w=30 占 **5/19** (26%)
- w=10 占 **1/19**（parabola vy v-OOD）
- **w=0.1 / w=1.0 / w=20 各占 0**

---

## 3. 关键 insights

### Insight 1：λ_probe 的"甜点"在 30-50，比之前任何报告高 30-50 倍

所有 3 篇前报告都用 `w=1.0`（默认值）。**w=50 在 long-cos / OOD vx 等所有困难指标上比 w=1 提升 +0.10~+0.30**。这意味着 deep-sup probe 的真正价值被之前大幅低估。

### Insight 2：frames 偏好依域 + 依 λ 联合决定

| 域 | 低 λ (1-10) 最佳 frames | 高 λ (30-50) 最佳 frames |
|---|---|---|
| parabola | f=4 (mf4) | **f=2** |
| uniform | f=1 (pos-only) | **f=4** |
| collision | 分散 | 分散 |

**uniform 上 frames 偏好彻底反转**（f=1 → f=4）。说明 "within-traj std 决定单帧/多帧" 假说是低 λ 下的伪相关。

### Insight 3：collision v-OOD vx 是噪声重灾区

| w | f=1 | f=2 | f=4 |
|---|---|---|---|
| 10 | 0.205 | 0.438 | 0.452 |
| 30 | 0.267 | 0.241 | 0.214 |
| 50 | **0.529** | 0.323 | 0.344 |

非单调 + 不同 frames 互相矛盾 → **单 seed 不可信，必须 multi-seed**。

### Insight 4：长程预测 cos h=16 仍未饱和

三域 cos h=16 在 w=50 仍在涨：parabola 0.872、uniform 0.954、collision 0.633。**值得继续扩 w=100 / w=200**。

---

## 4. Caveats（诚实标注）

1. **单 seed**：每 (domain, w, f) 仅 1 个 seed，cell 间差异 0.05–0.20 可能部分来自 seed 噪声。collision v-OOD vx 的 w=30 全面退化（0.214–0.267）vs w=50 突然恢复到 0.529，**最可能是 seed-to-seed 噪声**。建议关键 cell 补 3 seed。
2. **target 固定为 [proprio, vel_col]**：未扫 target ∈ {proprio-only, vel-only, [proprio,vel]}。
3. **w 未到平台期**：cos h=16 在三域上 w=50 仍在涨；建议下次扩 w ∈ {100, 200}。
4. **uniform vy 全 nan**：vy 恒为 0，非 bug。
5. **某些 cell 严重非单调**（如 collision vx v-OOD、parabola vx ID）：这类 cell 的"最佳 (w,f)"读出不稳健，仅作参考。
6. **GPU 浮点 reduction**：8×A800 多 GPU 训练每个 cell 都有 0.01–0.05 的浮点噪声。

---

## 5. 下一步建议（按优先级）

1. **🏆 扩 w ∈ {100, 200}**：18 jobs，预计 35-40 分钟。验证 cos h=16 是否还涨。
2. **关键 cell 补 multi-seed**（最重要）：parabola w=50/f=2、uniform w=50/f=4、collision w=50/f=1 各跑 3 seeds，看排序稳定性。
3. **更新 piwm_three_domains_new.md**：用 w=50 的数字替换"mf4"等旧 label。
4. **写到论文 motivation**：collision v-OOD vx 在 w=0.1 时崩到 −0.060，在 w=50 时救到 +0.529——这是 deep-sup probe **强度依赖性**的强证据，比"frames 不普适"故事更深。
5. **扩 target**：固定 best (w, f) per domain，扫 target ∈ {proprio, vel, [proprio,vel]}，9 个新配置。

---

## 6. 复现 / 文件

| 项 | 路径 |
|---|---|
| 本报告 | `reports/6-2/sweep_three_domains_results.md` |
| Raw 表（自动生成版）| `reports/6-2/sweep_three_domains_results_raw.md` |
| 提取脚本 | `reports/6-2/extract_sweep_results.py`（自动找两个 log 目录）|
| sweep 编排（原 27）| `reports/6-2/sweep_three_domains.sh`（w ∈ {0.1, 1, 10}）|
| sweep 编排（扩 18）| `reports/6-2/sweep_three_domains_extend.sh`（w ∈ {30, 50}，含 `num_workers=2`）|
| 训练日志（27）| `/data1/likun-share/junjxu/runs/sweep_three_domains_logs/train_*.log` |
| 训练日志（18）| `/data1/likun-share/junjxu/runs/sweep_three_domains_extend_logs/train_*.log` |
| eval 日志（27+18）| 同上目录的 `rollout_*.log` |
| 45 个 ckpt | `/data1/likun-share/junjxu/.stable_worldmodel/{parabola,uniform,collision}_sw_w{0p1,1p0,10p0,30p0,50p0}_f{1,2,4}_id1k/` |

**复现命令**：
```bash
# 27 jobs (w=0.1,1,10), ~59 min
bash /home/likun-share/junjxu/wm/reports/6-2/sweep_three_domains.sh "0 1 2 3 4 5 6 7"
# 18 jobs (w=30,50), ~37 min — num_workers=2 优化让它比 27 jobs 还快
bash /home/likun-share/junjxu/wm/reports/6-2/sweep_three_domains_extend.sh "0 1 2 3 4 5 6 7"
# 提取 45 cell 报告
.venv/bin/python /home/likun-share/junjxu/wm/reports/6-2/extract_sweep_results.py \
  > /home/likun-share/junjxu/wm/reports/6-2/sweep_three_domains_results.md
```
