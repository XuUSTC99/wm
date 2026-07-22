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
- **真实数据同样成立(2026-07-12,3 种子 solid)**:Physion++ 直训 **FR vs TF headline h64 nMSE = FR 0.141 vs TF 1.174(好 8.3×,3v3 区间零重叠,cos 0.89 vs 0.50)**;物理结构(struct/cons/consacc)长程全差于纯 FR(0.168–0.266 vs 0.141)。出处:[physionpp §3.8](../../physion/physionpp_ood_longhorizon.md)。
- **迁移也最好**:phyworld→Physion zero-shot 所有配置中 free-rollout 最高(0.603,唯一逼近 random 天花板 0.607 者)。出处:[transfer_improvement_report.md](../../physion/transfer_improvement_report.md)。
- **跨模型也成立(2026-07-16,第二个 JEPA 实例,3 种子)**:冻结 DINOv2-small + adapter(DINO-WM 风格)复跑同协议,**FR≫TF = uniform 1.39× / parabola 1.69× / collision 1.68× / Physion++ 3.99×**(区间零重叠);注入 30 格 **27/30 不优于 baseline**(与 LeWM Fig16 一致);REAL-emb 位置 ρ **0.951 > LeWM 0.899**(presence 更强)、黑盒旁路 ρ 0.951≈all192(旁路跨模型完好)。**唯一异常 uniform pw300(0.67×)经 shuffle+weakpin 双对照证明是加权正则、非物理**(随机目标同样 0.77×+ID 退化)。出处:[detail/cross_model_dinowm.md](detail/cross_model_dinowm.md)、`raw_data/runs/dinowm/`。
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
| collision | 0.393 | **0.172**(scale0.5+np20) | −56%,3 种子 0.172/0.168/0.177(**0.172±0.004**)✅✅ |

- 超过所有物理结构方法(uniform aug05 0.068 < 最好物理配置 0.109)。
- **交互非平凡**:appearance×np20 冲突(0.472 ❌)、scale×np20 最佳叠加(0.208 ✅)、app×scale 温和(0.262);三重组合更差(0.253)。强度甜点 0.5,0.7/1.0 过增广。
- **时序增广证伪**:temporal stride 治不了 v-OOD(0.556 还不如 app 的 0.376)→ v-OOD 是增广啃不动的硬骨头。

**真实域反转(论文的关键警示)**:
- Physion++ 直训:appearance 0.5 把 friction_collision nMSE 从 0.062 崩到 **6.44(~100×)**。⚠️ 两个精度点(2026-07-14 审计修正):① **friction 的 cos 是同步下降的(0.978→0.894),方向真实——它不是 cos 陷阱实例**;真正的 cos 陷阱是 **deform_clothhit(cos 0.610→0.913 升 +0.30 而 nMSE 0.772→3.69 崩)与 h64 整体(cos 0.794→0.870 升而 nMSE 0.280→0.311 退化)**,归 C6。② 100× 的幅度来自 per-scene nMSE(分母敏感,physionpp §3.6 已降级为参考);**方向坚实**:cos 降 + horizon 整体 h16 nMSE 2.5×(0.016→0.041)同向退化。出处:`physionpp/eval_pp_fr{,_app05}_e20.log`。
- Physion zero-shot 迁移:增广无用(0.597 < random 0.607)。
出处:[general_augmentation.md](../../6-24/final/general_augmentation.md)、physionpp 报告、[cross_dataset_ledger.md](../../6-24/final/cross_dataset_ledger.md)。

## C4. 物理结构先验在共享 latent JEPA 上全线失效(注入方式×init×域 全否)✅✅

> **claim 口径(信息论+建设性,2026-07-13 收敛)**:latent 已把物理**状态**编强(位置 ρ 0.9、冗余分布在黑盒 190 维,probe-190 实证)→ 再注入**同一份状态**是**冗余的**(不带新信号→不提升)、还占容量分梯度(→有害,intrinsic dim 塌 39–90%)。真正短板不在状态、在**预测器长程动力学**(可解码但 rollout 漂)——统一了"注入状态没用"与"free-rollout 有用"。物理注入要帮须补 latent **真正缺的**(动力学/守恒量),非已在里面的状态(⚠️未证实,future work)。**精确 scope**:限"往保留全维黑盒预测通道的共享 latent 嫁接状态";更高维 slot/更好映射仍是塞冗余状态、没堵旁路→同预测无用(reweighting 到 300× 已直接测"给物理更多梯度份额",只到持平);突破须改架构堵旁路(extrinsic),而 extrinsic 也只解决承重、不解决编码器-OOD(见下 PIWM 块)。详见 [why_physics_structure_fails 层3](detail/why_physics_structure_fails.md)。

**变体扫描(uniform/parabola/collision, pusht init, both-OOD nMSE,基线=纯FR)**:
| 变体 | 结果 | 出处 |
|---|---|---|
| 自由 accel MLP | uniform 0.131→**0.155** ❌ / parabola r/m 0.127→**0.178** ❌ / collision 0.393→**0.560** ❌(2026-07-12 从日志补齐三域) | rollout_{uniform_dyn_mlp,parabola_dyn_mlp,collision_structdyn}_fr_id1k.log |
| **严格 PIWM**(a=g 只学重力,parabola 上是正确物理!) | parabola r/m 0.127→0.173 ❌ / uniform 0.131→0.206 ❌ / **collision 0.393→0.559 ❌(日志补齐)**;干净分区下与自由 MLP 相当 | FINAL_SUMMARY §0.3 + rollout_collision_piwm_const_fr_id1k.log |
| 无标签物理(不钉真值,只要求 slot 按二阶动力学平滑演化、让位置自组织进 slot;动机=physion_collide 纯视频无 proprio) | uniform 0.171 / parabola 0.359 / collision 0.653,全掉 ❌ | physics_paper_design §5;名词详解见 [02 名词表](02_story_and_novelty.md) |
| grounded(上行的有标签对照:同结构 + slot 钉真值 proprio) | uniform 0.166 / parabola 0.392 ❌ —— 连完美标签+正确物理形式也伤 | 同上 |
| consistency loss(约束"怎么变"而非"是什么":预测 rollout 位置 slot 的差分速度 ≈ 真值速度,不假设加速度形式 → 本为 collision 冲量设计) | collision 0.57–0.64 全差于纯 FR 0.393 ❌,专治的域恰恰最差;**光滑域也败:uniform 0.151 vs 0.131 ❌、parabola r/m 0.147 vs 0.127 ❌** | optimization_plan + rollout_uniform_cons_B_v1 / parabola_structpos_cons1p0acc;实现 [train.py L181-211](../../../le-wm/train.py) |
| probe 单用(λ1,f2,[pos,vel]) | 0.131→0.167 ❌;pixel 同向:both-OOD 19.83 / h28 21.71(均低于 baseline 20.41/22.09) | 本 session 2×2(7-07;pixel 7-11 补全) |
| structpos 单用(无 pos_weight 加权) | 0.131→0.183 ❌ | kinematics_exploration |
| probe+structpos 组合(pos_weight 加权) | 0.125,打不过 structpos 单用 0.114;pixel 同向:both-OOD 20.02 / h28 21.91,低于 structpos 单用(21.30/22.41)甚至 baseline | 本 session 2×2(**latent+pixel 双尺闭环 ✅✅**) |
| 速度进 pos_weight 加权 slot([pos,vel]×pw30) | 物理量进 slot **整体有害**——uniform 0.114→**0.207** ❌❌、collision 0.621 ❌;唯一例外 parabola r/m 0.122→0.096(**小提升 −0.026**,三种子 0.093/0.091/0.104,量级很小、单域单分区),**可能**因速度在抛体里是线性驱动量、可外推——不当卖点。附:不同物理量进 slot(**只作可能原因**)——位置 structpos 随复杂度单调变差(0.132/0.160/0.596),加速度净效果 Δ=+0.075/−0.064/+0.025 大体对应速度在该域有无信息价值(数据点少,不拔高成规律) | Arm C + 2026-07-12 三种子;structpos 三域 rollout_{um,par,col}_structpos_fr_pw30 |
| **2026-07-12 补齐的 9 臂**(probe/组合/posvel 的 par+col 版、plain slot 两域、grounded col) | 全部就位:Table 2 凑满 30 格,28/30 不优于基线;仅有的两个"例外"均在 parabola r/m 单种子(probe 0.115=基线种子下沿=噪声;posvel 0.093 待种子判) | `/data1/.../runs/aaai_p0/rollout_{parabola,collision}_*` |

**pretrain vs post-train 2×2(2026-07-11,60ep 统一,证伪"要在预训练注入"假设)**:
| 域 | scratch+off | scratch+on(Δ) | pusht+off | pusht+on(Δ) |
|---|---|---|---|---|
| parabola | 0.559 | 0.678(+0.119) | 0.244 | 0.467(+0.223) |
| uniform | 0.349 | 0.576(**+0.227**) | 0.131@20ep | 0.166(+0.035) |
| collision | 0.359 | 0.675(**+0.316**) | — | — |

物理 from-scratch 也伤,uniform/collision 上伤得比嫁接更狠。出处:`/data1/likun-share/junjxu/runs/pretrain_physics/rollout_pp_*.log`、[EXPERIMENT_PLAN.md](../pretrain_physics/EXPERIMENT_PLAN.md)。⚠️ 单种子。

**干净基线定版(2026-07-11;lewm 120ep parabola + 本会话 60ep um/col;数字均已核 h28 无爆点)**:

**⚠️ 2026-07-22 三种子重跑,单种子值已作废**(12 run:3 域 × on/off × seed{1234,42},配已有 3072)。判决分区 nMSE,mean±std:

| 域 | 分区 | scratch_off | scratch_on | Δ_scratch (Welch 95%CI) | 判决 |
|---|---|---|---|---|---|
| **uniform** | both | 0.222±0.028 | **0.657±0.095** | **+0.435** [+0.276,+0.594] | 确证有害 |
| collision | both | 0.502±0.043 | 0.648±0.013 | **+0.147** [+0.074,+0.220] | 确证有害 |
| parabola | **r/m** | 0.476±0.169 | 0.447±0.063 | **−0.029** [−0.318,+0.260] | **持平(跨 0)** |

逐种子(3072/1234/42):uniform off 0.192/0.230/0.246、on 0.750/0.560/0.662;parabola off 0.343/0.420/0.666、on 0.375/0.494/0.473;collision off 0.538/0.513/0.454、on 0.635/0.661/0.649。源 `raw_data/runs/pretrain_physics/rollout_pp2_*_scratch_*_s{1234,42}.log` + 汇总 `SEED_RESULTS.md`。

- **headline 句(已改)**:from-scratch 共训**救不回**注入——uniform +0.435、collision +0.147 两域 CI 不跨 0 确证有害,parabola 持平。**⚠️ 旧稿的"三域 Δ 全为正 / all six cells positive"已被三种子推翻**(parabola 单种子 +0.03 → 三种子 −0.029),写作勿再用。论证不受损:堵死"要在预训练注入"这条辩护只需"从头共训不能让注入变好",**持平不是 rescue**。
- **⚠️ parabola from-scratch 基线极不稳**(0.343–0.666,std 0.169,跨度大于待测效应)→ 该域只能报持平,不能报增益。
- **⚠️ 勿用 parabola both-OOD 的 "+0.550 / 8.7×"**:scratch_on both=1.201 被 h28 爆点 197 万污染,判决走 r/m(详见 [NOTE_from_lewm_pretrain_caveat.md](NOTE_from_lewm_pretrain_caveat.md))。
- **副结论(喂 C6)**:120ep 降 LR 后 scratch 的 pred_loss 已收敛到 ~0.008(与 pusht 同量级)但 rollout OOD 仍差 2.8×(r/m 0.343 vs 0.124)→ **训练 loss 收敛 ≠ rollout 泛化好**。
- 出处:`/data1/.../runs/pretrain_physics/rollout_pp2_par_*.log`、`/data1/.../runs/aaai_p0/rollout_pp2_{um,col}_*.log`、[EXPERIMENT_PLAN.md §6.5-7](../pretrain_physics/EXPERIMENT_PLAN.md)。
**迁移侧同向**:物理结构越强迁移越差(pos_weight 加权 0.551 = 全场最差,< random 0.607)。

**机制解释(论文的"为什么")**:位置只占 2/192 维、pred_loss 平均后 ~1% 权重 → 黑盒 190 维冗余编码位置、预测靠会漂的黑盒通道;物理约束与黑盒 predictor 梯度打架(λ_probe=10 时 probe/pred 梯度比 15–125×,encoder intrinsic dim 塌 39–90%)。

**外部 baseline:官方 PIWM extrinsic 忠实移植到 phyworld(2026-07-13,回应"无外部方法 baseline",⚠️单种子)**:官方 code(VAE→extractor→已知方程 dynamics,extrinsic-conti,最有利 δ=0)三阶段全流程。**物理参数学对了**:uniform s=1.0/g=0(匀速)、parabola **g_y=−0.028 ≈ 真值 −0.025(重力)** → 移植忠实。**rolled-out 位置 ρ(vs LeWM free-rollout)**:

| 分区 | PIWM uniform | LeWM uniform | PIWM parabola | LeWM parabola |
|---|---|---|---|---|
| ID | **0.96** | 0.93 | **0.98** | 0.70 |
| **r/m-OOD** | 0.33 ⚠️ | **0.89** | 0.56 ⚠️ | **0.74** |
| v-OOD | **0.97** | 0.87 | **0.98** | 0.72 |
| **both-OOD** | 0.48 ⚠️ | **0.87** | 0.44 | **0.51** |

- **关键 nuance**:extrinsic PIWM **没有旁路问题**(物理态是架构强制的主通道、按构造承重),ID/v-OOD 甚至比 LeWM 更准;但 **r/m/both-OOD 仍崩**——崩因不是"slot 被绕过",而是**VAE 编码器扛不住没见过的球尺寸**(把 OOD 外观编错位置,dynamics 救不了坏初值)。→ **OOD 鲁棒与"承重"是两个正交问题:extrinsic 解决承重,但不解决编码器-OOD。**
- **对结论的影响(需 paper 定夺口径)**:这**部分温和了**"extrinsic 是唯一出路"的乐观——extrinsic 修好承重、拿到干净 ID 物理外推,但**买不到 OOD 鲁棒**(编码器瓶颈另说)。诚实框架:物理结构(intrinsic 旁路 / extrinsic 编码器)两条路都不买 OOD 鲁棒,真杠杆仍是训练协议(free-rollout)。
- ⚠️ 公平性:LeWM 侧 ρ 是 probe(eval 集 80/20 训)读 rolled-out latent,PIWM 是直接输出,协议偏袒 LeWM;即便如此 PIWM ID/v 仍更高、OOD 崩非协议造成。nMSE 两者归一化基不同不可比,只用 ρ。
- 出处:`/data1/likun-share/junjxu/runs/piwm_baseline/{redyn_*.log,eval_*_d0.json}`;移植码 `PIWM/phyworld_port/{train_piwm.py,eval_piwm.py}`(官方码 `PIWM/piwm-official/`);详见 [piwm_baseline/PLAN.md](../piwm_baseline/PLAN.md)、[NOTE_from_lewm_piwm_baseline.md](NOTE_from_lewm_piwm_baseline.md)、图 [figures/fig9_piwm_vs_lewm.png](figures/fig9_piwm_vs_lewm.png)。

## C5. pos_weight 加权:多数域×分区仍有害,只个别格子救回持平(无净增益)

- **全曲线 pw1→300 逐格(nMSE↓)**:加权到头,**4 个域×分区只有 2 个救回与 baseline 持平**——uniform·both-OOD(pw≥30 → 0.132 vs 0.136)、parabola·r/m(pw≥100 → 0.106 vs 0.122);另 2 个**全程有害、任何权重救不回**——uniform·r/m 0.21–0.26(baseline 0.14)、collision 0.59–0.69 且**越加越差**(baseline 0.393)。→ 加权只在 slot 占主导那格消掉危害,**多数格子仍负效果、无净增益**(uniform·both 到 pw300 无 U 形)。收窄口径 "only 2 of 4 cells reach parity; r/m & collision harm persist at all λ"。出处 [lbr_ablation §4](../lbr_ablation/PLAN.md)。
- 正向证据剩 **pixel 尺**(structpos_pw30 both-OOD 21.30 vs 20.41 dB)与 **v-OOD 解码位置 ρ 0.991/0.987**,但仍单种子(作正向 claim 需补种子,否则按"边际、指标依赖"写进 anatomy)。
- **⚠️ pos_weight 加权持平只在 uniform 成立(2026-07-12 日志补齐)**:parabola slot+pw30 r/m 0.160 vs 基线 0.127 ❌、collision 0.596 vs 0.393 ❌ —— 富动力学域连 pos_weight 加权后的 slot 都仍有害。论文 §4.5 已按此口径收窄("removes the harm where the slot matches the dynamics")。
- **pos_weight 加权+运动学 = 光滑域长程 pixel 净超纯 FR**:uniform pixel h28 **+1.25dB**、both-OOD **+1.26dB**(structcv_fr_pw100:23.34/21.67 vs 22.09/20.41);v-OOD 位置 ρ 全场最高 0.991/0.987;parabola both-OOD 0.313→**0.262**(accel 学到常重力)。
- **四条件缺一不可**:free-rollout + pos_weight + 光滑域(collision 失败)+ pixel 尺(latent 聚合被 190 维稀释看不出)。
- 软肋:r/m-OOD(外观变)posρ 仅 0.29→0.43(无运动学 0.96);且 pos_weight 加权伤 Physion 迁移。
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
- **init 消融 ✅✅(2026-07-14,3 种子)**:physion 直训 h64 nMSE = **scratch 0.038±0.003 < cube 0.065±0.021 ≈ pusht 0.080±0.016**。**scratch 显著最好**(error bar 与两者不重叠,预训练 init 域偏见拖累);**cube(3D) vs pusht(2D) 不显著**(error bar 重叠,单种子的"3D 好 17%"是假象——cube seed1234 跳到 0.094)。→ physion 直训"要不要 init"是真信号(不要最好),"init 域接近度"不是可靠杠杆;与 phyworld(pusht init > random)相反 → **init 价值随目标数据规模/真实度递减**。出处:[physionpp §3.8](../../physion/physionpp_ood_longhorizon.md)。
- 早期支撑:预训练域 > 规模 > task-FT(5.5M PushT ViT-tiny ≈ 749M ImageNet DiT-XL;vel_x 阶梯 pixel-stats 0.516→random 0.573→ImageNet 0.754→PushT 0.878→DiT 0.890);phyworld 上任何 SSL 微调 net-negative。

---

## 附:可信度警示

| 事项 | 状态 |
|---|---|
| 6-2 的 sweep_three_domains_results / piwm_three_domains_new | **作废**(init bug),不可引用 |
| 5-x 的早期 R² 数字 | 已被 5-26 协议修正推翻,只引修正后的 MSE+ρ |
| 多数 6-24/7-x 结果 | ⚠️ 单种子;C1/C4 已补种子/干净基线 ✅✅,C5 补后只到持平 |
| collision pixel 尺 | decoder-limited 不可信,collision 只用 latent 尺 |
| probe/structpos 2×2 的 pixel 尺 | ✅ 已补全(2026-07-11),latent+pixel 双尺闭环 |
| **parabola 的一切 both-OOD nMSE** | **弃用为判决指标**(h28 除零爆点,C6#4);判决走 r/m-OOD。**波及历史数字**:C2 的 np16 0.313→0.416、C3 的 aug 0.313→0.115、C5 的运动学 0.313→0.262 等 parabola both-OOD 结论,写论文前须用 r/m-OOD 复核方向(C3 已有 r/m 佐证 0.127→0.065 ✅;C2/C5 待复核,好在均非 headline) |
