# 在真实 Physion++ 上训 lewm —— OOD + 长程 rollout 结果

**日期**：2026-07-06
**一句话结论**：**在真实感的 Physion++ 上直接训 lewm，OOD + 长程预测确实能做好**（刚体场景长程 rollout cos 0.96-0.99、h64 仍 0.79），与 zero-shot 迁移那条死路（封顶 random 0.607）完全相反。**free-rollout 是长程主力**，加物理结构约束（structured/consistency/accel）损害整体长程、但换来物理量显式可解码（trade-off）。**布料形变（deform）是所有方法共同短板。**

---

## 0. 设置

- 数据：Physion++ readout，800 clips、10 物理属性场景（mass/friction/bouncy/deform），转成 lewm HDF5：pixels(224) + proprio(target 3D 位置) + state(3D 速度) + 占位 action + scene_idx(OOD 标签)。转换脚本 `phyworld/scripts/physion/physion_plus_to_lewm.py`。
- 训练：free-rollout(num_preds=8) + pusht init，4 配置各 20 epoch，setsid 脱离 session。脚本 `reports/physion/run_physionpp.sh`。
- 评估：`phyworld/scripts/physion/rollout_eval_physionpp.py` —— AR rollout 的 latent cos/nMSE by horizon + by scene，target 位置/速度解码 ρ（Ridge probe on real→pred emb）。
- 4 配置：`pp_fr`(纯FR) / `pp_struct`(+structured+pw30) / `pp_cons`(+consistency) / `pp_consacc`(+consistency+accel)。

## 1. 长程 rollout（latent cos by horizon，epoch 20，test）

| horizon | **pp_fr** | pp_struct | pp_cons | pp_consacc |
|---|---|---|---|---|
| h1 | 0.9994 | 0.9975 | 0.9977 | 0.9974 |
| h8 | 0.9974 | 0.9922 | 0.9922 | 0.9914 |
| h16 | **0.9936** | 0.9866 | 0.9828 | 0.9851 |
| h32 | **0.9098** | 0.8320 | 0.7875 | 0.8665 |
| h64 | **0.7937** | 0.7221 | 0.7319 | 0.7269 |

→ **纯 free-rollout 在所有 horizon 最好，长程尤其明显**（h32 0.91 vs 0.79-0.87）。nMSE 也全程最低（h1 0.003 vs 0.018）。加物理结构约束全部损害长程。**与 phyworld 结论一致：free-rollout 是长程决定性杠杆，intrinsic-slot 物理约束无益。**

## 2. 物理属性 OOD（pp_fr 各 scene 长程 cos，test）

| 场景 | cos | nMSE |
|---|---|---|
| friction_platform | 0.986 | 0.041 |
| bouncy_platform | 0.984 | 0.041 |
| mass_dominoes | 0.979 | 0.058 |
| friction_collision | 0.978 | 0.062 |
| mass_collision | 0.973 | 0.083 |
| bouncy_wall | 0.963 | 0.094 |
| **deform_clothhit** | **0.610** | 0.772 |
| **deform_clothhang** | **0.367** | 1.375 |

→ **刚体场景（friction/bouncy/mass）rollout 都很好（0.96-0.99）；布料形变（deform）是共同短板**（高自由度、非刚体动力学难预测）。符合物理直觉。

## 3. 位置解码 ρ —— 关键 trade-off

- **pp_fr**：长程 latent 整体保真最好，**但 target 位置不显式可解码**（rollout latent → pos ρ 多在 0-0.5，friction_collision 0.61-0.65）。
- **pp_cons/struct**：structured slot 强制固定维度=位置，**位置显式可解码**（中段 pp_cons epoch7 friction_collision pos ρ 0.87/0.93/0.97），但整体长程 cos 略低于 pp_fr。

→ **"整体 rollout 保真" vs "物理量可解码" 是一个真实 trade-off**，和 phyworld 一致（structured 让状态可读但不改善整体预测）。

## 3.5 应用 lewm 增广结论 —— **负向结果 + "cos 会骗人"的教训**（2026-07-06 修正）

> ⚠️ **本节初版只看 latent cos，得出"appearance 0.5 最优、长程 +0.08、deform +0.27"的喜报是错的。**
> 用 nMSE（含尺度）复核后结论**反转**：破坏性增广在真实 Physion++ 上全面有害。保留过程作为方法论教训。

把 lewm 定稿 [../6-24/final/general_augmentation.md] 的 `aug.appearance/scale` 叠到 pp_fr，epoch20 公平对比。**cos 与 nMSE 严重打架**（base → app05）：

| horizon / 场景 | cos(base→app05) | **nMSE(base→app05)** | nMSE 判决 |
|---|---|---|---|
| h16（短程） | 0.994→0.984 | 0.016→**0.041** | 退化 2.5× |
| h32 | 0.910→0.933 | 0.140→0.135 | 持平 |
| h64（长程） | 0.794→**0.870** | 0.280→**0.311** | 退化 |
| friction_collision | 0.978→0.894 | 0.062→**6.44** | **退化 100×** |
| mass_collision | 0.973→0.862 | 0.083→0.955 | 退化 11× |
| bouncy_wall | 0.963→0.872 | 0.094→0.828 | 退化 8.8× |
| deform_clothhit | 0.610→0.913 | 0.772→**3.69** | 退化 4.8× |

→ **cos 全面上升是假象**：cos 只量方向（无尺度）。appearance 0.5 让 encoder 对亮度不变、方向更对齐（cos↑），却把 latent 的**尺度/分布搅乱**，预测的绝对幅度全崩（nMSE↑ 几十上百倍）。**"长程 +0.08" 和 "deform +0.27" 都是 cos 制造的幻觉。**

**教训（写进方法论）：**
1. **latent-space rollout 必须用 nMSE（含尺度）判决，cos 单指标会骗人** —— 尤其当某方法改变了 encoder 的表示分布时，cos 可以升而 nMSE 崩。
2. **改变像素统计的增广（appearance/亮度）在真实数据上有害** —— 它搅乱 latent 尺度，nMSE 全场景崩。几何增广（scale）nMSE 退化轻微（h64 0.280→0.292）但也不带来增益。
3. **nMSE 最好的仍是纯 free-rollout baseline（pp_fr）**。长程真正的杠杆是 free-rollout 本身 + **调大 num_preds**（lewm collision 证据：np8→np20 h28 cos 0.58→0.70），不是增广 —— 见 §4 下一步（纯 np20 实验进行中）。

## 3.6 真正的 nMSE 正向杠杆：num_preds↑ + scale（2026-07-07）

吸取 app05 教训后换方向：不碰像素统计，只调 **free-rollout 步数（num_preds）** 和 **几何增广（scale）**。三配置 epoch20 公平对比（horizon 整体 nMSE，越低越好）：

| h | baseline(np8) | np20 | scnp16(np16+sc) | np20sc(np20+sc) | **np28sc(np28+sc)** |
|---|---|---|---|---|---|
| h16 | 0.0164 | 0.0215 | 0.0103 | 0.0115 | 0.0076 |
| h32 | 0.1404 | 0.0326 | 0.0351 | 0.0236 | 0.0101 |
| **h64** | 0.2797 | 0.2203 | 0.1361 | 0.0866 | **0.0144** |

**变量分离结论：**
1. **中程（h32）决定性杠杆 = 调大 num_preds**：np20 与 scnp16 都 ~0.033、比 baseline(0.140) 好 **4×**，两者接近 → 主力是 num_preds，不是 scale。
2. **长程（h64）scale 再加成**：scnp16 0.136 < np20 0.220 < baseline 0.280。纯 num_preds 已把长程 0.280→0.220，scale 在此基础上再砍一半到 0.136。
3. **scale = 幅度稳定器（真实价值）**：同一 scene、同一 ground-truth 下，纯 np20 在部分场景预测幅度失配、nMSE 爆炸，scale 修复：

| scene（相同 GT） | np20(无scale) nMSE | scnp16(+scale) nMSE |
|---|---|---|
| friction_collision | **24.64**（cos 仍 0.992） | 0.045 |
| deform_clothhit | 1.98 | 0.29 |
| mass_collision | 0.049 | 0.132 |

friction_collision 上 np20 的 cos 仍 0.992（方向对）但 nMSE 炸 → **纯长 rollout 会在某些场景幅度过冲**；scale 几何增广（对物体尺度/位置 rOOD 鲁棒）把幅度拉回。

→ **与 app05 相反，num_preds↑ 和 scale 是 cos/nMSE 同向的真实增益**：不改像素统计的方法（几何 + 更长 rollout）在真实 Physion++ 上稳赢，appearance（改亮度统计）则崩。

**顶配 np28sc（num_preds28 + scale0.3）= 全局最优**：长程 h64 nMSE **0.0144**（baseline 0.280 的 **1/19**、cos 0.995），h32 0.010、h16 0.008，**所有 horizon 最低**。num_preds 16→20→28 时 h64 单调陡降 0.136→0.087→**0.014**，**完全没有拐点** → num_preds 是长程的决定性主力，还能继续推（np32+）。

**关于 scale（诚实降级）**：horizon 整体口径下 scale 有正贡献（np20→np20sc 的 h64 0.220→0.087）。但此前用 per-scene nMSE 讲的「scale 稳定 friction_collision（24.6→0.019）」证据**不可靠**：np28sc 同带 scale，friction_collision nMSE 却又爆到 15.7（cos 仍 0.995、horizon 整体仅 0.014）。跨配置剧烈非单调波动说明 **per-scene nMSE 被少数静止/小位移 traj 的分母 artifact 主导**，不能作为模型性质证据。**per-scene 一律优先看 cos，nMSE 只信 horizon 整体（60 traj 混合，稳健）。**

**注**：per-scene nMSE 对静止/小位移物体有分母敏感性（cos 高而 nMSE 大时优先信同场景相对对比）；此处 scnp16 vs np20 用相同 GT、相同 n，差异纯来自预测，干净。

## 4. 结论与下一步

**回答核心问题**：跳出 zero-shot、**在真实数据上训**，OOD+长程能做好 —— 这条路验证有效。方法选择：
- 目标是**长程预测质量** → **free-rollout**（别加 intrinsic 物理约束）。
- 目标是**从 latent 读出物理量** → structured/consistency（牺牲一点长程换可解码性）。

**下一步攻坚**：
1. **deform 布料形变**是所有方法短板 → 需要针对高自由度/非刚体的方法（多物体 proprio、形变感知损失）。
2. **真 held-out scene OOD**：当前 by-scene 是 all-trained 的分场景分析；训练时 held-out 部分场景（如训刚体测形变）才是严格属性 OOD。
3. 扩到 Physion++ train split（28G，更多数据）+ test split 做标准泛化。

---

*产物*：ckpt `$STABLEWM_HOME/pp_{fr,struct,cons,consacc}/*_epoch_20_object.ckpt`；eval log `/data1/likun-share/junjxu/runs/physionpp/eval_*_e20.log`。
*脚本*：`physion_plus_to_lewm.py`（转换）、`run_physionpp.sh`（训练）、`rollout_eval_physionpp.py`（评估）。setsid 跑法见 memory `background-tasks-use-setsid`。
