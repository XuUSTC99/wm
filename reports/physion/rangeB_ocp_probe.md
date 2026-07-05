# Physion 评估 · 范式 B(zero-shot 迁移)· OCP 探针报告

**日期**: 2026-07-05
**结论一句话**: phyworld-collision 上训练的 lewm,zero-shot 迁移到 Physion OCP,**平均不优于未训练的随机 encoder**;仅在物理机制最接近的 Support / Collide 上有可信的小幅真实增益(+0.07~0.10 AUC)。→ zero-shot 跨 domain 迁移弱,需范式 A(在 Physion 上训)。

---

## 1. 目标与范式

评估当前 lewm(latent JEPA 世界模型)方法在 **Physion** 物理理解 benchmark 上的效果。
本报告为**范式 B**:直接用 phyworld 上训好的 lewm ckpt,**不重训**,在 Physion 上做
OCP(Object Contact Prediction,预测两物体最终是否接触)评估。

## 2. 数据

- **Physion Test-Core**(271 MB):8 场景 × 150 个 test trial 的 redyellow MP4(red=agent, yellow=patient)+ `labels.csv`(OCP 真值 True/False)。
- 来源:`cogtoolslab/physics-benchmarking-neurips2021`(S3)。
- 落盘:`/data1/likun-share/junjxu/physion_raw/`(注:大盘 /data2–7 当前账号无写权限,仅 /data1 可写)。
- 每场景 150 trial,类别接近平衡(如 Collide 75/75)。

## 3. 方法(快速探针)

- **模型**:`collision_rerun_w5p0_f2_id1k` epoch_20(phyworld collision 上训的 lewm)。
- **表征**:冻结 encoder → 前 45 帧(fps=30 → **1.5s**,符合 Physion OCP 协议)→ 每帧 `cls token → projector → 192维` → **meanstd 池化** → 384维特征。
- **分类**:`StandardScaler + LogisticRegression`,5-fold StratifiedKFold。
- **对照**:同一模型 encoder+projector 权重 **random-init**(未训练),跑同样探针 → 判断信号是"真物理"还是"架构/低级视觉"。
- 脚本:`phyworld/scripts/physion/ocp_probe.py`。

> ⚠️ **这不是 Physion 官方协议**。官方协议在独立的 *readout* 训练集上训分类器、在 *test* 集上测。本探针是 test-set 内 5-fold CV,只给方向性信号。官方数字见 §6(readout HDF5 下载中)。

## 4. 结果:trained vs untrained(random-init)

| 场景 | Trained AUC | Random AUC | 真实增益 Δ | 判定 |
|---|---|---|---|---|
| **Support** | 0.700 | 0.597 | **+0.10** | ✅ 真物理信号 |
| **Collide** | 0.632 | 0.562 | **+0.07** | ✅ 真物理信号 |
| Contain | 0.568 | 0.573 | −0.01 | ✗ 无增益 |
| Link | 0.555 | 0.606 | −0.05 | ✗ |
| Dominoes | 0.508 | 0.604 | −0.10 | ✗ trained 更差 |
| Drape | 0.563 | 0.657 | −0.09 | ✗ |
| Drop | 0.532 | 0.639 | −0.11 | ✗ |
| Roll | 0.521 | 0.412 | +0.11 | ⚠️ 假象(trained 仍≈chance,random 异常低) |
| **平均** | **0.572** | **0.581** | **−0.01** | |

(acc/AUC 均为 5-fold 均值;每场景 n=150,std 约 ±0.05~0.10,噪声较大。)

## 5. 分析与结论

1. **平均而言 zero-shot 无效**:trained 平均 AUC 0.572 ≈ random 0.581,collision 表征整体上并不比未训练 encoder 更能预测 Physion 接触。
2. **random baseline 本身就 >chance(0.56~0.66)**:未训练 ViT + redyellow 物体的空间/运动统计,已能"蹭"到部分接触信号 → 解读 trained 绝对值时必须扣掉这个架构先验。
3. **真信号只在 Support / Collide**:trained 绝对值高(0.63~0.70)且显著高于 random。这两个场景的物理(碰撞、支撑/稳定)与 phyworld collision 训练最接近 → 迁移最有效。
4. **Roll 的 +0.11 是假象**:trained 0.52 仍接近 chance,只是 random 异常低(0.41),不算 trained 的真实能力。
5. **domain gap 是主因**:phyworld(合成、简单渲染)→ Physion(TDW 真实感渲染、复杂场景),视觉分布差异大,zero-shot 迁移自然弱。

## 6. 局限 & 下一步

**局限**:
- 非官方协议(test-set CV),n=150 噪声大;
- 单一 ckpt(collision),未试 uniform/parabola;
- 池化/ctx 未调优(仅 meanstd、45帧)。

**下一步**:
- [进行中] **官方 readout 协议**:Collide readout HDF5(16G)下载中(~2.6h),用 readout 集训分类器、test 集测,给单场景标准数字。
- [计划] **范式 A**:在 Physion dynamics 集上训 lewm(需合成 action / 改无 action 训练),再评估 —— 这才是公平评估 lewm 方法在 Physion 上上限的方式。
- [可选] 多 ckpt 对照(uniform/parabola)、ctx/pool 调优、per-frame 时序 readout。

---

*脚本*: `phyworld/scripts/physion/{probe_smoke.py, ocp_probe.py}`
*复现*: `python phyworld/scripts/physion/ocp_probe.py --scenario Collide --ctx 45 --device cuda:1 [--random-init]`
