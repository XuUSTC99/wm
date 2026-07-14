# 实验计划：预训练 init 对 Physion++ 直训的影响（2D pusht vs 3D cube vs scratch）

**日期**：2026-07-12
**提出**：用户洞察——physion 是 3D，pusht init 是 2D pushing，域差大；lewm 官方发布的 **lewm-cube**（OGBench Cube = 机械臂堆方块，**3D 操作**）架构完全相同（ViT-tiny/14/224），更接近 physion，值得作为 init 对比 scratch。

---

## 0. 核心问题

**在真实 3D 物理数据（Physion++）上直训，预训练 init 的"域"重要吗？更接近的 3D init 是否胜过 2D pusht？预训练本身是否胜过 scratch？**

背景反直觉证据（ledger C7）：phyworld 上 init 阶梯 random 0.573 < ImageNet 0.754 < **PushT 0.878** < DiT 0.890——PushT（2D）反而赢 ImageNet，因 pusht 预训练带了"物体运动 predictor 先验"（与 2D/3D 无关）。**但那是合成 phyworld；physion 更真实/3D，cube 的 3D 操作先验可能真的更贴。这个实验直接验证。**

## 1. 三个 arm（同配置、同数据，只变 init）

| arm | init | 假设 |
|---|---|---|
| **pusht**（baseline） | `lewm_paper_pusht/weights.pt`（2D pushing） | 现状 |
| **cube**（主角） | `hf_cube/weights.pt`（3D OGBench 堆方块） | 3D 更接近 physion → 更好 |
| **scratch**（对照） | 无（random encoder） | 检验"预训练到底有没有用" |

- init 只加载 `encoder./projector./pred_proj.`（216 权重），`action_encoder` 一律 physion 自训 → cube 的 action_dim=25 不影响。
- init 加载有 guard（train.py:389：encoder <50% 加载即报错），防 silent random-init。

## 2. 固定配置

- **数据**：`physionpp_readout.h5`（完整 800 clip，10 场景）
- **模型配置**：**np20sc**（`wm.num_preds=20 aug.scale=0.3`，free-rollout）
  - 理由：接近最优（np20sc h64 nMSE 0.087，仅次于 np28sc 0.014），但比 np28（~11h）快（~7h）；长程足够长以显出 init 差异（init 效果主要在长程漂移上体现）。
  - 不选 np8：长程太短，init 差异被掩盖。不选 np28：太慢，3 arm 不划算。
- **epoch**：20（与全部现有结果一致，公平）
- **其余**：`+init_from_ckpt=<对应 weights>`（scratch arm 省略此项）

## 3. 评估

1. **主指标**：physion rollout horizon **nMSE + cos**（h1→h64），3 arm 同图对比。判据：nMSE 为准，cos 辅助（双指标交叉，防 §3.5/3.7 陷阱）。
2. **分场景**：刚体 vs deform（看 init 是否帮到某类场景）。
3. **可选加分**：held-out scene OOD（复用 htrain + `--group-scenes`）——cube init 是否泛化更好？这是最强论文点，但要多训 3 个 held-out arm（+~15h）。**先做主对比，有信号再加。**

## 4. 显存与 GPU 评估（⚠️ 待实测确认，下面是估算）

**当前 GPU（2026-07-12 查）**：各卡被占，free 仅 **22–28G**（GPU4 仅 3G）——**比之前紧张**。

| 配置 | 序列帧 | batch | 估算峰值显存 | 当前能否单卡塞下 |
|---|---|---|---|---|
| np20sc | 23 | 32 | ~28–35G | ❌ 勉强/超（free 22–28G） |
| np20sc | 23 | **24** | ~22–28G | ◐ 单张 free≥28G 的卡（GPU0/7）可 |
| np20sc | 23 | **16** | ~16–20G | ✅ 大多数卡可 |

**结论 & 建议**：
- 3 arm **并行**需 3 张 free>28G 的卡——当前不满足（只有 GPU0/7 ~28G）。
- 方案 A（推荐）：**batch=16 + 3 arm 并行**（每 ~18G，凑得出 3 张卡），代价训练略慢/略抖。
- 方案 B：**batch=24 顺序跑**（一次一张卡，~7h×3=21h）。
- 方案 C：**等 GPU 空**（之前有全空时段）再 batch=32 并行。
- ⚠️ 精确峰值需实测：我可跑一个 arm 到 step 50 读 nvidia-smi 峰值再定 batch（下一步）。

## 5. 时间估算

| 方案 | 墙钟 |
|---|---|
| 3 arm 并行（batch16，3 卡） | ~7h |
| 3 arm 顺序（batch24，1 卡） | ~21h |
| + held-out OOD 加分（×3 arm） | +~15h |

## 6. 预期结果与论文价值

- **若 cube > pusht**：证明"init 域越接近目标越好"，3D 操作先验帮到 3D 物理 → 支持用户直觉，是干净的正向消融。
- **若 pusht ≈ cube > scratch**：证明"预训练有用但域不敏感"，dynamics predictor 先验是关键（呼应 phyworld 的反直觉）。
- **若三者接近**：证明"physion 直训数据量足够，init 不关键"——也是有用的负向结论（弱化对预训练的依赖）。
- 任一结果都对论文有价值（init 消融是审稿人常问的）。⚠️ 单种子起步，有信号再补种子。

## 7. 执行清单（待用户确认后启动）

1. [ ] 实测单 arm 峰值显存 → 定 batch
2. [ ] 启动 3 arm（pusht / cube / scratch）× np20sc × epoch20，setsid
3. [ ] 训完各自 rollout eval（nMSE+cos by horizon + by scene）
4. [ ] 对比表 + 结论写入 physionpp 报告新 section + ledger
5. [ ] （可选）held-out OOD ×3 arm

---

*cube 权重*：`$STABLEWM_HOME/hf_cube/weights.pt`（69M，curl hf-mirror 下载，架构 = pusht 同构已验证）
*脚本*：复用 `run_physionpp.sh` / 手写 setsid（加 `+init_from_ckpt`）、`rollout_eval_physionpp.py`
