# 改造 phyworld 方法以提升 Physion 迁移 —— 探索报告

**日期**：2026-07-05
**一句话结论**：**phyworld → Physion 的 domain gap 是根本性的。zero-shot 迁移的天花板是 random 架构先验（Physion mean AUC ≈ 0.607），phyworld 上的任何方法改进（物理结构 / 外观增广 / 训练时长）都无法突破它。** 要在 Physion 上做好，必须跳出"phyworld 训 + zero-shot 迁移"框架（在真实数据上训/微调，或用真实视频预训练 encoder）。

---

## 0. 目标与评估

目标：把 lewm 会话在 phyworld 上的方法改进，迁移到真实感的 Physion/Physion++，提升 OCP（物体接触预测）准确率。

评估：`eval_physion_suite.py`，冻结 encoder → 前 1.5s 帧 → OCP 5-fold logistic。指标 Physion 8 场景 mean AUC。

**baseline（天花板参照）**：
- random-init（未训练 encoder）：**0.607**
- zero-shot collision（phyworld 训）：0.572

## 1. 尝试一：正向方法 ckpt 直接迁移

评估 lewm 会话训好的 collision 正向方法 ckpt：

| 方法 | Physion mean AUC |
|---|---|
| random | **0.607** |
| free-rollout（最轻） | 0.603 |
| +consistency+accel | 0.582 |
| +consistency | 0.566 |
| +pos_weight 承重 | 0.551 |

→ **全部 < random，且越加物理结构越差**。承重（phyworld both-OOD 最好的方法之一）迁移最差。

## 2. 尝试二：外观增广（appearance invariance）

假设：迁移差是合成 vs 真实的外观 gap；让 encoder 对外观不变应能缩小它。
实现：训练时 per-trial 亮度/对比度 jitter（`aug.appearance`，train.py）。

| aug 强度 | Physion mean AUC |
|---|---|
| 0（free-rollout） | 0.603 |
| 0.5 | 0.597 |
| 0.3 | 0.579 |

→ **假设证伪**。外观增广最多"趋近 baseline、少损害"，从不净提升。亮度/对比度抹不平真实 domain gap（更深在纹理/渲染/形状）。

## 3. 尝试三：epoch sweep（训练时长 vs 迁移）

假设："训得越多越把通用视觉训坏"，应当早停 / 少训。
方法：现成 free-rollout ckpt 的 epoch 1→20 直接评估（无需重训）。

| epoch | Physion mean AUC |
|---|---|
| 1 | 0.554 |
| 2 | 0.563 |
| 5 | 0.578 |
| 10 | 0.595 |
| 20 | 0.603 |
| random(0) | 0.607 |

→ **假设也证伪**，且方向相反：epoch 越多迁移**越好**、单调**逼近** random。真相不是"训练损害"，而是 **random 架构先验就是天花板，训练只能恢复到接近、超不过；训练不足或加结构反而偏离**。

## 4. 硬结论

三条独立证据（正向方法、aug、epoch）一致：

> **phyworld 学到的"物理"对 Physion OCP 没有超过随机架构先验的迁移价值。domain gap 根本性，zero-shot 迁移封顶在 random ≈ 0.607。**

phyworld 上把 OOD/长程做得再好，也不会自动迁移到真实感数据 —— 这不是方法问题，是"合成 → 真实"这堵墙。

## 5. 出路（超出当前框架）

- **在 Physion/真实数据上训练或微调** lewm（有监督或自监督），让 encoder 见真实分布。
- 换 **大规模真实视频预训练的 encoder**（如 V-JEPA / 真实数据 ViT）作 backbone，再上物理方法。
- 评估协议升级：官方 readout-split（Physion++ test 已下载可扩展），得到更严格的标准数字。

## 6. 这套评估 pipeline 的价值

正是它诚实地量出了这堵墙。任何新方法（lewm 会话的、或真实数据训的），一行命令 `eval_physion_suite.py --ckpt <ckpt>` 就能验证"到底迁移过去没有"，判据固定：**Physion mean AUC 要显著 > 0.607 才算真突破**。

---

*脚本*：`phyworld/scripts/physion/{eval_physion_suite,ocp_probe,physion_plus_probe}.py`；*增广*：`le-wm/train.py` 的 `aug.appearance`；*实验*：`reports/physion/run_aug.sh`。
*数据*：`eval_*.json` 存档在 `reports/physion/`。
