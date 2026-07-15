# raw_data —— 迁移用原始数据源(2026-07-15 从本机 `/data1` 收集)

> **为什么有这个目录**:原服务器即将停用。所有报告(尤其 [reports/7-11/aaai_paper/](../reports/7-11/aaai_paper/))引用的数据源原本散在 `/data1/likun-share/junjxu/` 下,**随服务器一起消失**。此目录把**全部 log / eval json / 权威 config** 收进 git 仓内,使论文数字**离开本机后仍可回溯、可重画图**。
>
> **1311 个文件 / 54 MB**。文件清单(路径+大小+mtime)见 [MANIFEST.tsv](MANIFEST.tsv)。

---

## 0. ⚠️ 用之前必读:三个种子的文件命名不一致

这是本仓最容易踩的坑(**已在 2026-07-15 迁移清点时发现并订正过引用**):

| 种子 | 文件命名 | 位置 |
|---|---|---|
| **3072**(**默认种子**) | **无 `_s` 后缀**,形如 `rollout_uniform_motion_baseline_fr_id1k.log` | `runs/structdyn_eval/` |
| 1234 | `..._s1234.log` | `runs/aaai_p0/` |
| 42 | `..._s42.log` | `runs/aaai_p0/` |

- **不存在 `rollout_*_baseline_fr_s3072.log` 这种文件**(phyworld 三域)。旧稿曾把出处写成 `_s{3072,1234,42}`,是错的,已改。
- **例外**:`runs/physionpp/` 下的 Physion++ 文件**确实**有 `_s3072` 后缀(`eval_pp_fr_s3072.log`),那里的三种子命名是齐的。
- 域名也不一致:`aaai_p0` 用 `uniform`,`structdyn_eval` 用 `uniform_motion`。

**复核过的三种子基线**(2026-07-15 从下列 log 重建,与论文 Fig 2 完全一致):

| 域·协议 | s3072 | s1234 | s42 | mean±std |
|---|---|---|---|---|
| uniform TF (both-OOD) | 0.3077 | 0.2971 | 0.2947 | **0.300±0.006** |
| uniform FR (both-OOD) | 0.1313 | 0.1457 | 0.1312 | **0.136±0.007** |
| parabola TF (r/m-OOD) | 0.4201 | 0.4116 | 0.4978 | **0.443±0.039** |
| parabola FR (r/m-OOD) | 0.1270 | 0.1254 | 0.1146 | **0.122±0.006** |
| collision TF (both-OOD) | 1.1136 | 1.2067 | 1.1368 | **1.152±0.040** |
| collision FR (both-OOD) | 0.3933 | 0.4947 | 0.5481 | **0.479±0.064** |

---

## 1. 目录结构

| 目录 | 内容 | 原位置 | 文件 |
|---|---|---|---|
| `runs/` | **全部训练/评估 log 与 eval json**,按原 run 目录结构保留 | `/data1/likun-share/junjxu/runs/` | 1013 |
| `configs/` | **每个 run 的权威超参** `config.yaml`,扁平化命名为 `<run名>.yaml` | `/data1/likun-share/junjxu/.stable_worldmodel/<run>/config.yaml` | 285 |
| `physion_eval/` | Physion OCP 评估结果 json(含 `eval_random_baseline.json` = 0.607 天花板) | 仓内 `reports/physion/*.json` 的副本 | 13 |

### `runs/` 各子目录用途(按论文相关度排序)

| 子目录 | 文件 | 用途 / 谁在引用 |
|---|---|---|
| **`structdyn_eval/`** | 410 | **主力**:30 格物理注入全扫 + LBR pos_weight 全曲线 + 默认种子(3072)基线。撑 [physics_injection_full_scan.md](../reports/7-11/aaai_paper/detail/physics_injection_full_scan.md)、[load_bearing_reweighting.md](../reports/7-11/aaai_paper/detail/load_bearing_reweighting.md) |
| **`aaai_p0/`** | 64 | 种子 1234/42 的基线与 structpos/posvel 臂;**probe REAL-emb ρ 的出处**(`rollout_{域}_baseline_fr_s1234.log` 的 `probe applied to REAL embs` 段)。撑 storyline 步1、Fig 1 |
| **`physionpp/`** | 67 | Physion++ 直训(TF/FR 三种子、num_preds 消融、增广)。撑 Fig 2 真实数据 8.3×、Fig 7、Fig 4 |
| **`pretrain_physics/`** | 36 | 预训练 2×2(scratch/pusht × physics on/off),撑"从头共训 Δ+0.558" |
| **`piwm_baseline/`** | 11 | 官方 PIWM 移植的训练/评估(`eval_{uniform_motion,parabola}_d0.json`)。撑 Fig 9 |
| **`consistency_eval/`** | 6 | consistency 臂(30 格里的一族) |
| `6-24_rerun_logs/` | 20 | 6-24 λ sweep 重跑(pred_loss 翻案那批) |
| `aug_eval/`, `decoder_viz/`, `pixel_*`, `lambda_sweep_*` | ~60 | 增广评估、解码可视化、pixel/PSNR 与 λ sweep |
| `sweep_*_logs`, `6-2_*`, `control_fixinit_logs`, `frozen_*` | ~250 | 早期 sweep 与诊断 |
| `*_BROKEN/` | ~120 | **⚠️ 数值不可引用**(init 静默丢 192 key,见 ledger)。保留仅作存档 |

---

## 2. 论文核心数字 → 源文件速查

| 论点 | 数字 | 源 |
|---|---|---|
| **位置可解码(presence)** | ρ 0.80–0.96 (both-OOD) | `runs/aaai_p0/rollout_{uniform,parabola,collision}_baseline_fr_s1234.log` → 段 `probe applied to REAL embs` |
| **rollout 崩(not use)** | collision h28 cos 0.24 | 同上 `--- ... vs horizon ---` 段 |
| **黑盒旁路 probe-190** | ρ 0.78–0.92 | `runs/structdyn_eval/` 的 probe190 段 |
| **30 格全扫** | 25 差/4 平/1 赢 | `runs/structdyn_eval/rollout_*_id1k.log` + `runs/aaai_p0/`(逐臂对照表见 [physics_injection_full_scan §5](../reports/7-11/aaai_paper/detail/physics_injection_full_scan.md)) |
| **三种子复核(4 个 † 格)** | 见 full_scan ⁿ 脚注 | `runs/structdyn_eval/rollout_{parabola_probeF2_fr,uniform_probeF2_structpos_pw30_fr}_s{1234,42}.log` |
| **LBR pos_weight 全曲线** | pw1→300 | `runs/structdyn_eval/rollout_{域}_structpos_fr_pw*_id1k.log` |
| **free-rollout 2.2–3.6×** | 见 §0 表 | 三种子:`structdyn_eval`(3072) + `aaai_p0`(1234/42) |
| **Physion++ 8.3×** | TF 1.174 → FR 0.141 | `runs/physionpp/eval_pp_{tf,fr}_s{3072,1234,42}.log` |
| **长 rollout 1/19** | h64 0.280→0.014 | `runs/physionpp/eval_pp_fr_{np20,np20sc,np28sc}_e20*.log` |
| **从头共训 Δ+0.558** | pretrain 2×2 | `runs/pretrain_physics/rollout_pp*_{um,par,col}_*.log` |
| **PIWM 对照** | ρ 0.33 vs 0.89 | `runs/piwm_baseline/eval_{uniform_motion,parabola}_d0.json` |
| **迁移天花板 0.607** | pos_weight 0.551 最差 | `physion_eval/eval_random_baseline.json` 等(逐配置表见 [real_data_physion §1](../reports/7-11/aaai_paper/detail/real_data_physion.md)) |
| **cos 陷阱** | cos 1.50 而 nMSE 0.21 | `runs/physionpp/eval_pp_fr{,_app05}_e20.log` |

---

## 3. 日志怎么读

`rollout_*.log` 的关键段:

```
--- cos / nMSE vs horizon ---        # 逐 horizon 的 cos 与 nMSE
  both-OOD   n= 1189  cos=+0.9410  nMSE=0.1457     # 分区聚合行 ← 判决数字取这里
[baseline] probe applied to REAL embs:            # 真实帧编码的可解码性(presence)
  REAL both-OOD   pos0ρ=+0.955  pos1ρ=+0.862 ...
  PRED both-OOD   pos0ρ=+0.894 ...                # rollout 预测 latent 的可解码性(not use)
```

**判读口径**(与论文一致):uniform/collision 取 **both-OOD**;**parabola 取 r/m-OOD**——其 both-OOD 在 h28 有除零爆点(球出框→目标方差→0→nMSE 飙百万),详见 [evaluation_traps §2](../reports/7-11/aaai_paper/detail/evaluation_traps.md)。

---

## 4. 没有收进来的东西(体积原因)——如需请另行迁移

| 数据 | 大小 | 原位置 | 说明 |
|---|---|---|---|
| **模型 ckpt** | **448 G** | `/data1/likun-share/junjxu/.stable_worldmodel/<run>/` | 291 个 run 的权重。**超参已存 `configs/`,可据此重训**;权重本体未搬 |
| Physion 原始数据集 | 29 G | `/data1/likun-share/junjxu/physion_raw/` | **公开数据集**,可重新下载 |
| PhyWorld 原始数据集 | 735 M | `/data1/likun-share/junjxu/phyworld_raw/` | `{collision,parabola,uniform_motion}_{30K,eval}.hdf5`,**公开数据集**,可重下 |
| decoder_viz 解码图 | 594 M | `/data1/likun-share/junjxu/runs/decoder_viz/` | 解码可视化 png(log 已收)。属**派生产物**,可由 ckpt 重生成 |
| lorabaseline | 170 G | `/data1/likun-share/junjxu/lorabaseline/` | 与本论文无关 |

> ⚠️ **`STABLEWM_HOME` 环境变量当前指向 `/data1/likun-share/.stable_worldmodel`(空目录)**;真实 ckpt 库在 **`/data1/likun-share/junjxu/.stable_worldmodel`**(注意多一层 `junjxu/`)。迁移后若要重训,记得重设此变量。

---

## 5. 复现/重画图

- 图脚本:[reports/7-11/aaai_paper/figures/storyline_figures.py](../reports/7-11/aaai_paper/figures/storyline_figures.py) —— **数字是内嵌硬编码的**(不读 log),故**离开本机也能一键重画**;要改数就改脚本顶部的数组。
- 各图的数据表+源:[detail/figures_gallery.md](../reports/7-11/aaai_paper/detail/figures_gallery.md)。
- 总账:[01_results_ledger.md](../reports/7-11/aaai_paper/01_results_ledger.md)。
