# 结果总账 —— 按论文主张归类

每条主张(C1~C7)下列出支撑数字、出处报告、可信度。**⚠️ = 单种子**;✅✅ = 多种子或多域交叉验证。
指标约定:latent **nMSE↓**(both-OOD 为最难分区);pixel **PSNR dB↑**;ρ = 解码 Pearson 相关。

---

## C1. free-rollout(去 teacher forcing)是唯一跨合成/真实通用的主升力 ✅✅

LeWM 原文用 num_preds=1 的单步 teacher forcing;改为自回归多步 free-rollout(np8):

**3 种子定版(2026-07-11 P0-1,seeds 3072/1234/42,mean±std;parabola 判决用 r/m-OOD,见下方爆点规则)**:

| 域 | 判决分区 | teacher-forced | free-rollout | 差距 |
|---|---|---|---|---|
| uniform | both-OOD | 0.300±0.006(0.308/0.297/0.295) | **0.136±0.007**(0.131/0.146/0.131) | **2.2×,区间零重叠** |
| parabola | **r/m-OOD** | 0.443±0.047(0.420/0.412/0.498) | **0.122±0.006**(0.127/0.125/0.115) | **3.6×** |
| collision | both-OOD | 1.153±0.040(1.114/1.207/1.137) | **0.479±0.064**(0.393/0.495/0.548) | **2.4×** |

- 三域、两模式、三种子,FR 与 TF 的区间完全不重叠 → **headline 多种子成立 ✅✅**。长程 cos 普遍 +0.3。出处:[piwm_dynamics_conclusion.md](../../6-24/piwm_dynamics_conclusion.md) §3.3 + `/data1/.../runs/aaai_p0/rollout_*.log`。
- **⚠️ parabola both-OOD nMSE 弃用(2026-07-11 lewm 交接)**:六个 parabola 臂 h28 nMSE 全部爆点(3.2万~101万;球出框→目标方差→0→除零,h28 cos 却正常 0.55~0.95),both-OOD 聚合被污染。parabola 一律以 **r/m-OOD** 判决(实测无爆点);旧 both-OOD 数字(TF 0.750±0.027 / FR 0.279±0.035)仅作方向参考。uniform/collision h28 正常(~1–2),不受影响。
- **真实数据同样成立**:Physion++ 直训,free-rollout 是长程主力(h32 cos:fr 0.91 vs 加物理结构 0.79–0.87)。出处:[physionpp_ood_longhorizon.md](../../physion/physionpp_ood_longhorizon.md)。
- **迁移也最好**:phyworld→Physion zero-shot 所有配置中 free-rollout 最高(0.603,唯一逼近 random 天花板 0.607 者)。出处:[transfer_improvement_report.md](../../physion/transfer_improvement_report.md)。
- 5-27 rollout 报告提供了"为什么":num_preds=1 时 1-step cos 0.98–0.99 但多步漂移,漂移速度=f(动力学复杂度) uniform<parabola<collision——teacher forcing 掩盖误差累积。

## C2. 训练 rollout 长度要匹配动力学复杂度(horizon-complexity matching)✅✅

| 域 | 最优 num_preds | 证据 |
|---|---|---|
| collision(冲量) | **≈20** | both-OOD 0.393→0.294(−25%),h28 cos 0.58→0.70,ID 3× |
| uniform / parabola(光滑) | **8** | np16 反而有害:uniform h28cos 0.969→0.814;parabola both 0.313→0.416 |
| Physion++(真实) | **≥28↑** | 中程好 baseline 4×;顶配 np28sc h64 nMSE 0.014(baseline 0.280 的 1/19);16→20→28 单调降、**未见拐点**(真实动力学比合成域更吃长 rollout) |

出处:[optimization_plan.md](../../6-24/final/optimization_plan.md) §3、physionpp 报告。collision np20 有 3 种子(0.208/0.198/0.244 配 scale)✅✅。

## C3. 数据增广 = 合成域最强 OOD 杠杆,但**不跨域、伤真实数据** ✅✅(反转部分)

**合成域(phyworld)**:
| 域 | 基线 both | 最优增广 | 降幅 |
|---|---|---|---|
| uniform | 0.131 | **0.068**(app0.5) | −48% |
| parabola | 0.313 | **0.115**(app0.5) | −63% |
| collision | 0.393 | **0.172**(scale0.5+np20) | −56%,**scale0.5 自己的 3 种子 = 0.172/0.168/0.177(0.172±0.004)✅✅**;旧文"~0.21"其实是 scale0.3 的种子(0.208/0.198/0.244),之前张冠李戴,2026-07-12 审计修正 |

- 超过所有物理结构方法(uniform aug05 0.068 < 最好物理配置 0.109)。
- **交互非平凡**:appearance×np20 冲突(0.472 ❌)、scale×np20 最佳叠加(0.208 ✅)、app×scale 温和(0.262);三重组合更差(0.253)。强度甜点 0.5,0.7/1.0 过增广。
- **时序增广证伪**:temporal stride 治不了 v-OOD(0.556 还不如 app 的 0.376)→ v-OOD 是增广啃不动的硬骨头。

**真实域反转(论文的关键警示)**:
- Physion++ 直训:appearance 0.5 把 friction_collision nMSE 从 0.062 崩到 **6.44(100×)**,cos 却在涨(cos 陷阱实例)。
- Physion zero-shot 迁移:增广无用(0.597 < random 0.607)。
出处:[general_augmentation.md](../../6-24/final/general_augmentation.md)、physionpp 报告、[cross_dataset_ledger.md](../../6-24/final/cross_dataset_ledger.md)。

## C4. 物理结构先验在共享 latent JEPA 上全线失效(注入方式×init×域 全否)✅✅

**变体扫描(uniform/parabola/collision, pusht init, both-OOD nMSE,基线=纯FR)**:
| 变体 | 结果 | 出处 |
|---|---|---|
| 自由 accel MLP | uniform 0.131→**0.155** ❌ / parabola r/m 0.127→**0.178** ❌ / collision 0.393→**0.560** ❌(2026-07-12 从日志补齐三域) | rollout_{uniform_dyn_mlp,parabola_dyn_mlp,collision_structdyn}_fr_id1k.log |
| **严格 PIWM**(a=g 只学重力,parabola 上是正确物理!) | parabola r/m 0.127→0.173 ❌ / uniform 0.131→0.206 ❌ / **collision 0.393→0.559 ❌(日志补齐)**;干净分区下与自由 MLP 相当 | FINAL_SUMMARY §0.3 + rollout_collision_piwm_const_fr_id1k.log |
| 无标签物理(不钉真值,只要求 slot 按二阶动力学平滑演化、让位置自组织进 slot;动机=physion_collide 纯视频无 proprio) | uniform 0.171 / parabola 0.359 / collision 0.653,全掉 ❌ | physics_paper_design §5;名词详解见 [02 名词表](02_story_and_novelty.md) |
| grounded(上行的有标签对照:同结构 + slot 钉真值 proprio) | uniform 0.166 / parabola 0.392 ❌ —— 连完美标签+正确物理形式也伤 | 同上 |
| consistency loss(约束"怎么变"而非"是什么":预测 rollout 位置 slot 的差分速度 ≈ 真值速度,不假设加速度形式 → 本为 collision 冲量设计) | collision 0.57–0.64 全差于纯 FR 0.393 ❌,专治的域恰恰最差;**光滑域也败(日志补齐):uniform 0.151 vs 0.131 ❌、parabola r/m 0.147 vs 0.127 ❌**(旧 both-OOD 0.291"看似变好"是爆点假象,又一 C6 实例) | optimization_plan + rollout_uniform_cons_B_v1 / parabola_structpos_cons1p0acc;实现 [train.py L181-211](../../../le-wm/train.py) |
| probe 单用(λ1,f2,[pos,vel]) | 0.131→0.167 ❌;pixel 同向:both-OOD 19.83 / h28 21.71(均低于 baseline 20.41/22.09) | 本 session 2×2(7-07;pixel 7-11 补全) |
| structpos 单用(无承重) | 0.131→0.183 ❌ | kinematics_exploration |
| probe+structpos 组合(承重上) | 0.125,打不过 structpos 单用 0.114;pixel 同向:both-OOD 20.02 / h28 21.91,低于 structpos 单用(21.30/22.41)甚至 baseline | 本 session 2×2(**latent+pixel 双尺闭环 ✅✅**) |
| 速度进承重 slot([pos,vel]×pw30) | uniform 0.114→**0.207** ❌❌;collision 0.621 ❌;**parabola r/m 0.093 ⚠️ 单种子"疑似正向"(基线种子区间 0.115–0.127 之下),种子复跑进行中,且跨域符号翻转不构成鲁棒性主张** | 本 session Arm C + 2026-07-12 补齐 |
| **2026-07-12 补齐的 9 臂**(probe/组合/posvel 的 par+col 版、plain slot 两域、grounded col) | 全部就位:Table 2 凑满 30 格,28/30 不优于基线;仅有的两个"例外"均在 parabola r/m 单种子(probe 0.115=基线种子下沿=噪声;posvel 0.093 待种子判) | `/data1/.../runs/aaai_p0/rollout_{parabola,collision}_*` |

**pretrain vs post-train 2×2(2026-07-11,60ep 统一,证伪"要在预训练注入"假设)**:
| 域 | scratch+off | scratch+on(Δ) | pusht+off | pusht+on(Δ) |
|---|---|---|---|---|
| parabola | 0.559 | 0.678(+0.119) | 0.244 | 0.467(+0.223) |
| uniform | 0.349 | 0.576(**+0.227**) | 0.131@20ep | 0.166(+0.035) |
| collision | 0.359 | 0.675(**+0.316**) | — | — |

物理 from-scratch 也伤,uniform/collision 上伤得比嫁接更狠。出处:`/data1/likun-share/junjxu/runs/pretrain_physics/rollout_pp_*.log`、[EXPERIMENT_PLAN.md](../pretrain_physics/EXPERIMENT_PLAN.md)。⚠️ 单种子。

**干净基线定版(2026-07-11;lewm 120ep parabola + 本会话 60ep um/col;数字均已核 h28 无爆点)**:

| 域 | 分区 | scratch_off(干净) | scratch_on | Δ_scratch | Δ_pusht(对照) |
|---|---|---|---|---|---|
| **uniform(headline)** | both | 0.192(脏基线 0.349→修好) | **0.750** | **+0.558** | +0.035 |
| collision | both | 0.538 | 0.635 | +0.097 | — |
| parabola | **r/m** | 0.343 | 0.375 | +0.03 | +0.07 |

- **headline 句**:物理在 from-scratch 下伤得更狠——uniform 上 Δ 从后训练嫁接的 +0.035 放大到 **+0.558**(基线越干净、物理伤越明显);三域干净设定全部 Δ>0,"物理要在预训练注入"强/弱假设全灭。C4 定稿 ✅✅。
- **⚠️ 勿用**:parabola both-OOD 的 "+0.550 / 8.7×"(scratch_on both=1.201 被 h28 爆点 197 万污染,详见 [NOTE_from_lewm_pretrain_caveat.md](NOTE_from_lewm_pretrain_caveat.md));之前"与 uniform +0.558 惊人一致"是巧合,已作废。
- **副结论(喂 C6)**:120ep 降 LR 后 scratch 的 pred_loss 已收敛到 ~0.008(与 pusht 同量级)但 rollout OOD 仍差 2.8×(r/m 0.343 vs 0.124)→ **训练 loss 收敛 ≠ rollout 泛化好**。
- 出处:`/data1/.../runs/pretrain_physics/rollout_pp2_par_*.log`、`/data1/.../runs/aaai_p0/rollout_pp2_{um,col}_*.log`、[EXPERIMENT_PLAN.md §6.5-7](../pretrain_physics/EXPERIMENT_PLAN.md)。
**迁移侧同向**:物理结构越强迁移越差(pos_weight 承重 0.551 = 全场最差,< random 0.607)。

**机制解释(论文的"为什么")**:位置只占 2/192 维、pred_loss 平均后 ~1% 权重 → 黑盒 190 维冗余编码位置、预测靠会漂的黑盒通道;物理约束与黑盒 predictor 梯度打架(λ_probe=10 时 probe/pred 梯度比 15–125×,encoder intrinsic dim 塌 39–90%)。

## C5. "承重"是物理编码唯一的正向 niche —— **3 种子后降级:latent 增益在噪声内,幸存的是 pixel/可解码性**

- **⚠️ 种子定版(2026-07-11 P0-4)**:structpos_fr_pw30 三种子 both-OOD = 0.114/0.135/0.147(**0.132±0.014**)vs baseline_fr **0.136±0.007** —— **latent 尺上的"净超"消失,单种子 0.114 vs 0.131 是种子噪声**。
- 幸存的诚实表述:**pos_weight 把 structpos 从净负救回与 baseline 持平**(pw1=0.183 有害 → pw30≈0.132 平),即"承重让物理 slot 无害共存",而非净提升;正向证据剩 **pixel 尺**(structpos_pw30 both-OOD 21.30 vs 20.41 dB;承重+运动学 h28 +1.25dB)与 **v-OOD 解码位置 ρ 0.991/0.987** —— 但这些仍单种子(若写进论文作为正向 claim,需补 pixel 臂种子;或按"边际、指标依赖"的口径写进 anatomy)。
- **⚠️ 承重持平只在 uniform 成立(2026-07-12 日志补齐)**:parabola slot+pw30 r/m 0.160 vs 基线 0.127 ❌、collision 0.596 vs 0.393 ❌ —— 富动力学域连承重后的 slot 都仍有害。论文 §4.5 已按此口径收窄("removes the harm where the slot matches the dynamics")。
- **承重+运动学 = 光滑域长程 pixel 净超纯 FR**:uniform pixel h28 **+1.25dB**、both-OOD **+1.26dB**(structcv_fr_pw100:23.34/21.67 vs 22.09/20.41);v-OOD 位置 ρ 全场最高 0.991/0.987;parabola both-OOD 0.313→**0.262**(accel 学到常重力)。
- **四条件缺一不可**:free-rollout + pos_weight + 光滑域(collision 失败)+ pixel 尺(latent 聚合被 190 维稀释看不出)。
- 软肋:r/m-OOD(外观变)posρ 仅 0.29→0.43(无运动学 0.96);且承重伤 Physion 迁移。
出处:[kinematics_exploration.md](../../6-24/kinematics_exploration.md)。⚠️ 单种子。

## C6. 评测方法论:四个系统性陷阱(方法论贡献)✅✅

1. **cos 陷阱(结构性)**:cos/K4-ρ 是 probe/structured loss 的对偶量,随 λ 单调涨是数学必然。实锤反转:probe 长程 latent cos 最稳(h28 0.882 vs 0.843)但 pixel 最差(19.93 vs 20.64);app 增广 cos 升而 nMSE 崩 100×。**判决必须用 nMSE/pixel,cos 永不单用。**
2. **zero-shot 迁移天花板 = random 架构先验(0.607)**:训练只能恢复到接近、超不过;三条独立证据(物理方法全 <random / 增广不破 / epoch 越多越逼近)。出处:transfer_improvement_report。
3. **协议混淆可制造假阴性**:random-init × 单帧 probe × with-projector 三个 confound 乘性叠加,uniform vx 从假阴性 0.166 修到 0.939(paper-init + K=4 + no-projector);另有 init 静默丢 192 key 使 45-config sweep 全作废(⚠️ 6-2 sweep 数值不可引用)。出处:5-26 negtive_result_report、diagnostic_report。
4. **nMSE 自身的除零爆点(与 #1 方向相反,别混写)**:parabola h28 附近个别轨迹球出框 → 目标 latent 方差→0 → nMSE 除零飙到 3 万~197 万,而同一 horizon 的 cos 仍 0.55~0.95 正常;both-OOD 聚合被这几条轨迹拉爆(六个 parabola 臂全中招)。**规则:引 nMSE 前先查 by-horizon 是否发散;parabola 判决走 r/m-OOD(各域无爆点)。** 两个方向合起来的论文表述:"cos 无尺度会漏报、nMSE 有尺度会被退化分母引爆——必须逐分区、逐 horizon 双指标交叉验证"。附:**训练 loss 收敛 ≠ rollout 泛化好**(scratch 120ep pred_loss 收敛到与 pusht 同量级 0.008,rollout r/m-OOD 仍差 2.8×)。出处:[NOTE_from_lewm_pretrain_caveat.md](NOTE_from_lewm_pretrain_caveat.md)、EXPERIMENT_PLAN §6.5。

## C7. 真实数据:zero-shot 死路 vs 直训活路 ✅✅

> **术语速览**(不然下面看不懂):
> - **Physion++** = 真实感物理仿真视频数据集(球碰撞/多米诺/布料/斜面滑动…,带真实 3D 位置+速度标注),画面接近真实,比 phyworld 的简化小球复杂得多。
> - **zero-shot 迁移** = 在 phyworld(简化合成)上训好模型 → **不在真实数据上训**,直接搬去 Physion 评估。测"合成里学的物理能否搬到真实"→ **结论:不能**(域差太大)。
> - **直训**(直接训练)= 跳过迁移,**直接把 Physion++ 真实视频喂给模型从头训**。测"真实数据上到底能做多好"→ **结论:很好**(长程/OOD 都做到了)。

- zero-shot phyworld→Physion:封顶 0.607(random 先验),仅 Support(+0.10)/Collide(+0.07)有真信号。
- **Physion++ 直训**(活路,统一以 nMSE 判——cos 无尺度会漏报,见 C6):长程 rollout **h64 nMSE 从 baseline 0.280 → 顶配 np28sc(np28+scale0.3)0.014(1/19)**;num_preds 16→20→28 单调陡降 0.136→0.087→0.014、**未见拐点**。刚体场景准(cos 0.96–0.99)、布料形变是短板。真 held-out scene OOD 结果见下条。
- 共同短板:布料形变(deform_clothhit 0.610 / clothhang 0.367)。
- **真 held-out scene OOD ✅(2026-07-12,⚠️单种子)**:训练**整场景排除**(bouncy_wall/deform_clothhang/mass_waterpush,各留同属性训练伙伴),rollout 泛化**分层**——mass_waterpush cos **0.972**(几乎不降,vs full 0.994,刚体质量动力学可迁移)、bouncy_wall 0.846(部分,vs 0.996)、deform_clothhang **0.263**(崩,vs 0.982,形变短板叠 OOD)。**可迁移的是刚体表观动力学,形变迁不动。** GROUP 混合 nMSE h64=3.22 是 deform+静止分母 artifact 假象,以 cos 分场景判(双指标交叉,呼应 C6)。出处:[physionpp §3.7](../../physion/physionpp_ood_longhorizon.md)。
- 早期支撑:预训练域 > 规模 > task-FT(5.5M PushT ViT-tiny ≈ 749M ImageNet DiT-XL;vel_x 阶梯 pixel-stats 0.516→random 0.573→ImageNet 0.754→PushT 0.878→DiT 0.890);phyworld 上任何 SSL 微调 net-negative。

---

## 附:可信度警示

| 事项 | 状态 |
|---|---|
| 6-2 的 sweep_three_domains_results / piwm_three_domains_new | **作废**(init bug),不可引用 |
| 5-x 的早期 R² 数字 | 已被 5-26 协议修正推翻,只引修正后的 MSE+ρ |
| 多数 6-24/7-x 结果 | ⚠️ 单种子;C1/C4 已补种子/干净基线 ✅✅,C5 补后降级 |
| collision pixel 尺 | decoder-limited 不可信,collision 只用 latent 尺 |
| probe/structpos 2×2 的 pixel 尺 | ✅ 已补全(2026-07-11),latent+pixel 双尺闭环 |
| **parabola 的一切 both-OOD nMSE** | **弃用为判决指标**(h28 除零爆点,C6#4);判决走 r/m-OOD。**波及历史数字**:C2 的 np16 0.313→0.416、C3 的 aug 0.313→0.115、C5 的运动学 0.313→0.262 等 parabola both-OOD 结论,写论文前须用 r/m-OOD 复核方向(C3 已有 r/m 佐证 0.127→0.065 ✅;C2/C5 待复核,好在均非 headline) |
