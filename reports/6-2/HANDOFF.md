# Handoff — LeWM × PhyWorld 物理世界模型 probing/deep-supervision 项目

**生成**：2026-06-04 | **给**：在 likun-A500 上接手的 Claude
**用法**：先读这份(1-2 页就能接上)。完整对话原始记录在同目录 `*.jsonl(.gz)`，一般不用读。

---

## 0. ⚠️ 路径 & 环境(最容易踩的坑)

- 项目根目录原叫 `agent_memory`，**已重命名为 `am`** → 所有路径是 `~/am/wm/...`（`/home/qlib/am/wm/...`）。**A500 上路径/用户名可能不同，先确认实际位置。**
- **Python 环境**：LeWM 用它自己的 venv `~/am/wm/le-wm/.venv/`（py3.10, torch 2.9.1+cu128, diffusers, peft, hydra, stable-pretraining/worldmodel）。默认 shell 的 `qlib_env`（`~/envs/qlib_env/bin/python`）**缺 einops 等，不能跑 LeWM**。
- **GPU 进程屏蔽家目录(用户偏好)**：启动训练用**相对路径** `cd ~/am/wm/le-wm && CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u train.py ...`，别用绝对 venv 路径(会在 `ps` 里暴露 `/home/qlib`)。**不要**用 `activate`(被 qlib_env 抢 PATH)或 `exec -a`(破坏 python 启动)。
- **共享 GPU**：A6000 48GB，多人共用。别往 util 99% 的卡上塞。
- 数据/ckpt 在 `~/.stable_worldmodel/`。

---

## 1. 项目是什么

把 **PhyWorld**(物理 benchmark：uniform_motion 匀速 / collision 碰撞 / parabola 自由落体；每域分 ID + 3 种 OOD 分区)的数据喂给 **LeWM**(Yann LeCun 组的 JEPA 像素世界模型，arXiv:2603.19312，本地 repo `~/am/wm/le-wm`)，研究它的 encoder/predictor 是否"理解物理"。也对照 DiT-XL。设计灵感来自 **PIWM**(arXiv:2412.12870，本地 `~/am/wm/PIWM`)。

**LeWM 损失** = `pred_loss`(latent 空间预测下一帧 emb 的 MSE) + 0.09·`sigreg_loss`(高斯正则防坍缩)。`history_size=3, num_preds=1`(**只训 1 步预测**)。我们加了第三项可选 deep-supervision probe loss(下文)。

---

## 2. 已完成的实验 + 关键结论

报告都在 `~/am/wm/reports/`。**真相源**：`5-26/negtive_result_report.md`。

1. **probe 协议/指标修正(5-26)**：初版"OOD encoder 崩溃"是假象，根因 = R² + K=1 单帧 + 只在 ID fit 三重 bug。换成 **MSE + Pearson ρ + K=4 多帧 MLP probe**(对齐 LeWM Table 1)后，表征在所有 partition 一致好(ρ≥0.74)。

2. **ID-only FT，真 ID→OOD(5-26 §6.4)**：用官方 `*_30K` 纯 ID 数据 FT(之前的 "leak-free" 其实 partition 泄漏)。结论：**LeWM FT 的真 ID→OOD 增益≈0**(之前的 +0.02 是 partition memorization)；**DiT LoRA FT 三域全灾难性净负**。

3. **AR rollout(5-26 rollout_results.md)**：用 ARPredictor 自回归。**1 步预测极准(cos 0.98-0.99)，但多步漂移、OOD 更快崩**。"能编码当前状态 ≠ 能遵守物理律演化"。

4. **PIWM deep-supervision(5-26 piwm_deepsup_results.md + 6-2/piwm_three_domains.md)**：在 FT loss 加 linear probe 监督物理量(recipe 来自 arXiv:2504.03861)。配置开关在 `le-wm/config/train/lewm.yaml` 的 `loss.probe.{weight,target,frames}`：
   - `weight=0` = 关(无需 enabled 开关，刚改的)；`target` 可为 `proprio` 或 `[proprio,action]`；`frames=1` 单帧 / `>1` 多帧窗口(最大=num_steps=4)。
   - **三域核心结论(piwm_three_domains.md)：多帧监督不普适！**
     - parabola：**mf4(多帧)最好**
     - uniform：**pos-only(单帧)最好**，多帧反而最差
     - collision：混合(ID→mf4, OOD→单帧)
   - **机制(有数据支撑)**：最佳监督粒度由**被监督速度量的"轨迹内 std"**决定 —— uniform 速度恒定(std=0)→ 多帧零新信息；parabola 的 vy 因重力在变(std=0.23)→ 多帧有用；collision vx 撞击跳变(std=0.21)→ 混合。已排除"数据采样不均"(三域 vx 范围都 [0.10,0.40])。

5. **LeWM 有像素 decoder 吗**：本体没有(纯 JEPA)。论文 App.D 有个**仅用于可视化**的 transformer decoder(从 192 维 CLS 解码，196 query token + cross-attn → 224×224)，repo 没开源。

---

## 3. 当前待办(优先级排序)

1. **λ_probe × frames sweep(用户要做，卡满了没启)**：parabola 上 grid `weight∈{0.1,1,10} × frames∈{1,2,4}`，target 固定 `[proprio,action]`。脚本已写好：`~/am/wm/reports/6-2/sweep_parabola.sh "<gpu列表>"`。**先用 1 配置 GPU smoke(1 epoch)验证新代码(weight>0 gate + frames=2)再全跑**。问题：frames 结论对 λ 鲁棒吗？uniform 上低 λ 会不会让多帧翻盘(验证"过度约束"假说)？
2. **最有潜力的研究方向**(详 `6-2/idea-stage/IDEA_REPORT.md`)：**🏆 PhysConsist-Rollout** —— 冻结 encoder，用 within-traj 方差从 latent **自动发现守恒量**(不靠已知方程)，rollout 每步把预测 latent **投影回守恒流形**，修长程 OOD 漂移。Novelty vs PIWM(不需已知方程)/HNN(latent+发现的)/Observer Effect 2602.12218(给修法非诊断)。**下一步先 /novelty-check 复核**。
3. ARPredictor rollout 可视化(给上面 decoder 解码 rollout latent，看 OOD 漂移在像素上长啥样)——proposal 在 `5-26/arpredictor_rollout_proposal.md`。

---

## 4. Novelty 现状(重要,别白做)

- deep-supervision 方法本身 = arXiv:2504.03861，**零创新**。
- "adaptation 损伤 latent physics" 已被 **arXiv:2602.12218 (Observer Effect)** 发了(≈我们 5-26 主结论)。
- 2026 这个 physics-probing-world-model 方向**极挤**(2602.07050 / 2603.20327 等)。
- 唯一较新的种子：**within-traj 方差 → 监督粒度**(6-2)+ **守恒投影 rollout**(idea-stage)。要做成论文必须升级成"带理论 + 跨域 + 真实视频"的预测性判据/方法，单靠 3 个 PhyWorld 玩具域不够。

---

## 5. 关键文件地图

| 路径 | 内容 |
|---|---|
| `~/am/wm/README.md` | 项目桥接文档 + 报告索引 |
| `~/am/wm/reports/5-26/negtive_result_report.md` | **主报告**(真相源)|
| `~/am/wm/reports/5-26/{rollout_results,piwm_deepsup_results,arpredictor_rollout_proposal}.md` | rollout / deep-sup / rollout 提案 |
| `~/am/wm/reports/6-2/piwm_three_domains.md` | **三域统一对比**(deep-sup 不普适 + within-traj std 机制)|
| `~/am/wm/reports/6-2/idea-stage/IDEA_REPORT.md` | 创新方向(PhysConsist-Rollout)|
| `~/am/wm/reports/6-2/{sweep_parabola.sh,run_piwm*.sh,run_evals.sh}` | sweep / 训练编排脚本 |
| `~/am/wm/le-wm/train.py` | LeWM 训练(含 `loss.probe` deep-sup)|
| `~/am/wm/le-wm/config/train/lewm.yaml` | 配置(probe 开关在这)|
| `~/am/wm/phyworld/scripts/rollout_eval_id1k.py` | rollout + K=1/K=4 probe 评估(`--ckpt`/`--tag` 切模型)|
| `~/.stable_worldmodel/` | 数据(`phyworld_*_id1k.h5`)+ ckpt(`*_paperinit_id1k/`, `*_piwm_*_id1k/`)|

---

## 6. 给接手 Claude 的第一步建议

1. 确认 A500 上项目路径(可能不是 `~/am/wm`)+ venv 可用
2. 读 `6-2/piwm_three_domains.md` 和 `6-2/idea-stage/IDEA_REPORT.md`(最新进展 + 方向)
3. 若继续实验：先跑 §3.1 的 sweep(脚本现成)；若推进研究：先 /novelty-check 复核 PhysConsist-Rollout
