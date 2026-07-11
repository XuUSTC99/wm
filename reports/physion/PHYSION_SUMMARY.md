# Physion / Physion++ 上评估 lewm 世界模型 —— 总结报告

**日期**：2026-07-08
**范围**：本 session 全部 Physion / Physion++ 工作的定稿汇总（评估 pipeline + zero-shot 迁移 + 在真实数据上训练 + OOD/长程杠杆）。
**分报告**：路线 A 详见 [transfer_improvement_report.md](transfer_improvement_report.md)；路线 B 详见 [physionpp_ood_longhorizon.md](physionpp_ood_longhorizon.md)；OCP 协议见 [rangeB_ocp_probe.md](rangeB_ocp_probe.md)。

---

## TL;DR

1. **zero-shot「phyworld 训 + 迁移 Physion」是死路**：domain gap 根本性，OCP 迁移封顶在 random 架构先验（mean AUC ≈ **0.607**），phyworld 上任何物理方法/增广/训练时长都突破不了。
2. **在 Physion++ 上直接训 lewm，OOD + 长程能做好**：长程 rollout（h64）nMSE 从 baseline **0.280 → 0.014（好 19×）**，主力是**调大 num_preds**（16→28 单调陡降、未见拐点）+ free-rollout + 几何 scale。
3. **一个差点骗过我们的陷阱**：appearance（亮度）增广让 latent cos 全面上升、看似大涨，但 nMSE 全场景崩（friction 100×）。**latent world-model 评估必须以 nMSE 为准，cos 单指标会骗人。**
4. **共同短板**：布料形变（deform）所有配置都难（clothhang nMSE ~0.8），是下一步攻坚点。

---

## 1. 背景与两条路线

- **模型**：lewm = latent JEPA 世界模型（ViT-tiny encoder → 192-d latent → AR predictor），free-rollout 训练。
- **两个数据集**：
  - **Physion**（OCP 物体接触预测）：冻结 encoder → 前 1.5s 帧 → readout 分类，指标 8 场景 mean AUC。
  - **Physion++**（真实物理状态）：pkl 里有 target 物体 3D 位置/速度 + 10 个物理属性场景（mass/friction/bouncy/deform），转成 lewm HDF5 后可直接训 + rollout 评估。
- **两条路线**：
  - **A**：phyworld 上训好 → zero-shot 迁移 Physion（省事，但……）。
  - **B**：直接在 Physion++ 真实数据上训（本 session 的主攻）。

---

## 2. 路线 A：zero-shot 迁移 —— 死路（封顶 random 0.607）

三条独立证据一致指向「domain gap 是墙」（Physion mean AUC，越高越好，random=0.607 是天花板参照）：

| 证据 | 结果 | 结论 |
|---|---|---|
| **正向方法 ckpt 迁移** | free-rollout 0.603 / +consistency 0.566 / +承重 0.551，**全部 < random 0.607** | 越加物理结构迁移越差 |
| **appearance 增广** | 0.3→0.579、0.5→0.597，**从不净提升** | 亮度抹不平真实 domain gap |
| **epoch sweep** | epoch 1→20：0.554→0.603，单调**逼近但不超** random | 训练只能恢复到接近架构先验，超不过 |

> **硬结论**：phyworld 学到的「物理」对 Physion OCP 没有超过随机架构先验的迁移价值。要在真实数据上做好，必须跳出 zero-shot 框架 —— 于是有了路线 B。

---

## 3. 路线 B：在 Physion++ 上直接训 —— 成功

### 3.1 free-rollout 是长程基础主力

4 配置（pp_fr 纯FR / +structured / +consistency / +consistency+accel）epoch20 对比，**纯 free-rollout 在所有 horizon 最好**（h32 cos 0.91 vs 加结构的 0.79-0.87）。加 intrinsic 物理结构约束（structured/consistency）**损害整体长程**，但换来物理量显式可解码（position slot 可 readout）—— 这是一个真实 trade-off，与 phyworld 结论一致。

### 3.2 属性 OOD：刚体好，形变是短板

pp_fr 各场景长程 cos：刚体场景（friction/bouncy/mass）**0.96-0.99** 都很好；布料形变 **deform_clothhit 0.61 / deform_clothhang 0.37** 是共同短板（高自由度非刚体动力学）。

### 3.3 陷阱与纠错：appearance 增广的 cos 假象

把 lewm 会话的 appearance 增广（aug.appearance=0.5）叠到 pp_fr，**latent cos 全面上升**（h64 0.79→0.87、deform 0.61→0.91），一度被误判为「最优增广」。**用 nMSE 复核后结论反转**：

| 场景 | baseline nMSE | app05 nMSE | 倍数 |
|---|---|---|---|
| friction_collision | 0.062 | 6.44 | **退化 100×** |
| bouncy_wall | 0.094 | 0.828 | 退化 8.8× |
| h64（整体） | 0.280 | 0.311 | 退化 |

根因：cos 只量方向（无尺度），appearance 让 encoder 对亮度不变、方向更对齐（cos↑），却搅乱 latent 尺度、绝对幅度全崩（nMSE↑）。**「长程 +0.08」是 cos 幻觉。**

### 3.4 变量分离与顶配：num_preds↑ + scale 才是真杠杆

换方向——不碰像素统计，只调 **free-rollout 步数（num_preds）** 和 **几何增广（scale）**。epoch20 公平对比 horizon 整体 nMSE（越低越好）：

| horizon | baseline(np8) | np20 | scnp16(np16+sc) | np20sc(np20+sc) | **np28sc(np28+sc)** |
|---|---|---|---|---|---|
| h16 | 0.0164 | 0.0215 | 0.0103 | 0.0115 | 0.0076 |
| h32 | 0.1404 | 0.0326 | 0.0351 | 0.0236 | 0.0101 |
| **h64** | 0.2797 | 0.2203 | 0.1361 | 0.0866 | **0.0144** |

**变量分离结论**：
1. **中程（h32）主力 = num_preds**：np20/scnp16 都好 baseline **4×**，两者接近 → 功劳在 num_preds 不在 scale。
2. **长程（h64）scale 再加成**：np20sc 0.087 < scnp16 0.136 < np20 0.220 < baseline 0.280。
3. **scale 有正贡献，但 per-scene 证据不可靠**：horizon 整体口径下 scale 确有增益（np20→np20sc 的 h64 0.220→0.087）。但 per-scene nMSE 被少数静止/小位移 traj 的**分母 artifact 主导**、跨配置剧烈非单调波动（friction_collision：np20 24.6 / np20sc 0.019 / np28sc 又 15.7，而 cos 全 >0.99）→ **per-scene 一律看 cos，nMSE 只信 horizon 整体**。

→ **与 appearance 相反，num_preds↑ 和 scale 是真实增益**：不改像素统计的方法（几何 + 更长 rollout）稳赢，改亮度统计的 appearance 崩。**全局最优 = np28sc（h64 nMSE 0.0144、cos 0.995、好 baseline 19×）**；num_preds 16→20→28 时 h64 单调陡降 0.136→0.087→**0.014**、**完全未见拐点** → num_preds 是长程决定性主力，可继续推 np32+。

---

## 4. 方法论教训（写进 memory）

> **latent-space world-model 的 rollout 评估必须用 nMSE（含尺度）判决，cos 单指标会骗人** —— 尤其当某方法改变了 encoder 的表示分布（尺度/范数）时，cos 可以升而 nMSE 崩。这次 appearance 增广就差点把「搞坏模型」报成「提升」。per-scene nMSE 对静止/小位移物体有分母敏感性（cos 高 nMSE 大时优先信同 GT 相对对比）。

---

## 5. 短板与下一步

1. **deform 布料形变**：所有配置共同短板（clothhang nMSE ~0.8），需形变感知损失 / 多物体 proprio。
2. **真 held-out scene OOD**：当前 by-scene 是「全训练后分场景分析」，非严格属性 OOD；训刚体、测形变才是真 OOD，论文说服力最强。
3. **Physion++ train split（28G）**：更多数据 + 官方 readout-split 标准泛化数字。
4. **num_preds 上限**：h64 未见拐点，np28→np32 继续推。

---

## 6. 产物与 pipeline 价值

- **一键评估**：`eval_physion_suite.py --ckpt <ckpt>`（OCP，判据固定：mean AUC 要显著 > 0.607 才算真突破）；`rollout_eval_physionpp.py --ckpt <ckpt>`（Physion++ 长程 nMSE/cos + 属性 OOD）。
- **数据转换**：`physion_plus_to_lewm.py`（Physion++ pkl → lewm HDF5，target=位置 std 最大物体）。
- **训练**：`run_physionpp.sh <GPU> <NAME> <sw> <cw> <pw> <app> <scale> <np>`，free-rollout + pusht init。
- 这套 pipeline 的价值：**诚实地量出了 domain gap 这堵墙，也量出了在真实数据上训的真实增益** —— 任何新方法一行命令就能验证「到底做好没有」。

---

*脚本*：`phyworld/scripts/physion/{eval_physion_suite,rollout_eval_physionpp,physion_plus_to_lewm,ocp_probe,physion_plus_probe}.py`
*ckpt*：`$STABLEWM_HOME/pp_fr_{fr,np20,scnp16,np20sc,np28sc}/*_epoch_20_object.ckpt`
*eval log*：`/data1/likun-share/junjxu/runs/physionpp/eval_*.log`
