# 结果总账 —— 按论文主张归类

每条主张(C1~C7)下列出支撑数字、出处报告、可信度。**⚠️ = 单种子**;✅✅ = 多种子或多域交叉验证。
指标约定:latent **nMSE↓**(both-OOD 为最难分区);pixel **PSNR dB↑**;ρ = 解码 Pearson 相关。

---

## C1. free-rollout(去 teacher forcing)是唯一跨合成/真实通用的主升力 ✅✅

LeWM 原文用 num_preds=1 的单步 teacher forcing;改为自回归多步 free-rollout(np8):

| 域 | teacher-forced | free-rollout | 降幅 |
|---|---|---|---|
| uniform | 0.308 | **0.131** | −57% |
| collision | 1.114 | **0.393** | −65% |
| parabola | 0.786 | **0.313** | −60% |

- 长程 cos 普遍 +0.3;三域一致 → 已设默认。出处:[piwm_dynamics_conclusion.md](../../6-24/piwm_dynamics_conclusion.md) §3.3。⚠️ 单种子(P0 要补 3 种子)。
- **真实数据同样成立**:Physion++ 直训,free-rollout 是长程主力(h32 cos:fr 0.91 vs 加物理结构 0.79–0.87)。出处:[physionpp_ood_longhorizon.md](../../physion/physionpp_ood_longhorizon.md)。
- **迁移也最好**:phyworld→Physion zero-shot 所有配置中 free-rollout 最高(0.603,唯一逼近 random 天花板 0.607 者)。出处:[transfer_improvement_report.md](../../physion/transfer_improvement_report.md)。
- 5-27 rollout 报告提供了"为什么":num_preds=1 时 1-step cos 0.98–0.99 但多步漂移,漂移速度=f(动力学复杂度) uniform<parabola<collision——teacher forcing 掩盖误差累积。

## C2. 训练 rollout 长度要匹配动力学复杂度(horizon-complexity matching)✅✅

| 域 | 最优 num_preds | 证据 |
|---|---|---|
| collision(冲量) | **≈20** | both-OOD 0.393→0.294(−25%),h28 cos 0.58→0.70,ID 3× |
| uniform / parabola(光滑) | **8** | np16 反而有害:uniform h28cos 0.969→0.814;parabola both 0.313→0.416 |
| Physion++(真实) | **20** | np20 中程好 baseline 4×;np20+scale 顶配 h64 nMSE 0.087(baseline 0.280 的 1/3.2) |

出处:[optimization_plan.md](../../6-24/final/optimization_plan.md) §3、physionpp 报告。collision np20 有 3 种子(0.208/0.198/0.244 配 scale)✅✅。

## C3. 数据增广 = 合成域最强 OOD 杠杆,但**不跨域、伤真实数据** ✅✅(反转部分)

**合成域(phyworld)**:
| 域 | 基线 both | 最优增广 | 降幅 |
|---|---|---|---|
| uniform | 0.131 | **0.068**(app0.5) | −48% |
| parabola | 0.313 | **0.115**(app0.5) | −63% |
| collision | 0.393 | **0.172**(scale0.5+np20) | −56%,3 种子均值 ~0.21 ✅✅ |

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
| 自由 accel MLP | uniform 上 accel 过拟合(真 a=0 学出 ~0.5·\|v\|),净贡献≤0 | piwm_dynamics_conclusion |
| **严格 PIWM**(a=g 只学重力,parabola 上是正确物理!) | parabola 0.313→0.372 ❌ / uniform 0.131→0.206 ❌,比自由 MLP 还差 | FINAL_SUMMARY §0.3 |
| 无标签物理(动力学先验自组织) | uniform 0.171 / parabola 0.359 / collision 0.653,全掉 ❌ | physics_paper_design §5 |
| grounded(有 proprio 接地) | uniform 0.166 / parabola 0.392 ❌ | 同上 |
| consistency loss(form-free 速度约束) | collision 0.57–0.64 全差于纯 FR 0.393 ❌ | optimization_plan |
| probe 单用(λ1,f2,[pos,vel]) | 0.131→0.167 ❌;pixel 同向:both-OOD 19.83 / h28 21.71(均低于 baseline 20.41/22.09) | 本 session 2×2(7-07;pixel 7-11 补全) |
| structpos 单用(无承重) | 0.131→0.183 ❌ | kinematics_exploration |
| probe+structpos 组合(承重上) | 0.125,打不过 structpos 单用 0.114;pixel 同向:both-OOD 20.02 / h28 21.91,低于 structpos 单用(21.30/22.41)甚至 baseline | 本 session 2×2(**latent+pixel 双尺闭环 ✅✅**) |
| 速度进承重 slot([pos,vel]×pw30) | 0.114→**0.207** ❌❌ | 本 session Arm C |

**pretrain vs post-train 2×2(2026-07-11,60ep 统一,证伪"要在预训练注入"假设)**:
| 域 | scratch+off | scratch+on(Δ) | pusht+off | pusht+on(Δ) |
|---|---|---|---|---|
| parabola | 0.559 | 0.678(+0.119) | 0.244 | 0.467(+0.223) |
| uniform | 0.349 | 0.576(**+0.227**) | 0.131@20ep | 0.166(+0.035) |
| collision | 0.359 | 0.675(**+0.316**) | — | — |

物理 from-scratch 也伤,uniform/collision 上伤得比嫁接更狠。出处:`/data1/likun-share/junjxu/runs/pretrain_physics/rollout_pp_*.log`、[EXPERIMENT_PLAN.md](../pretrain_physics/EXPERIMENT_PLAN.md)。⚠️ 单种子。
**迁移侧同向**:物理结构越强迁移越差(pos_weight 承重 0.551 = 全场最差,< random 0.607)。

**机制解释(论文的"为什么")**:位置只占 2/192 维、pred_loss 平均后 ~1% 权重 → 黑盒 190 维冗余编码位置、预测靠会漂的黑盒通道;物理约束与黑盒 predictor 梯度打架(λ_probe=10 时 probe/pred 梯度比 15–125×,encoder intrinsic dim 塌 39–90%)。

## C5. "承重"是物理编码唯一的正向 niche(窄但成立)✅(pixel 双指标)

- **pos_weight≈30 让 structpos 从净负翻净正**:both-OOD 0.131→**0.114**,ID/v-OOD 全面小胜,不带 r/m 崩(pw1=0.183 有害,pw100=0.136 回退,甜点 30)。
- **承重+运动学 = 光滑域长程 pixel 净超纯 FR**:uniform pixel h28 **+1.25dB**、both-OOD **+1.26dB**(structcv_fr_pw100:23.34/21.67 vs 22.09/20.41);v-OOD 位置 ρ 全场最高 0.991/0.987;parabola both-OOD 0.313→**0.262**(accel 学到常重力)。
- **四条件缺一不可**:free-rollout + pos_weight + 光滑域(collision 失败)+ pixel 尺(latent 聚合被 190 维稀释看不出)。
- 软肋:r/m-OOD(外观变)posρ 仅 0.29→0.43(无运动学 0.96);且承重伤 Physion 迁移。
出处:[kinematics_exploration.md](../../6-24/kinematics_exploration.md)。⚠️ 单种子。

## C6. 评测方法论:三个系统性陷阱(方法论贡献)✅✅

1. **cos 陷阱(结构性)**:cos/K4-ρ 是 probe/structured loss 的对偶量,随 λ 单调涨是数学必然。实锤反转:probe 长程 latent cos 最稳(h28 0.882 vs 0.843)但 pixel 最差(19.93 vs 20.64);app 增广 cos 升而 nMSE 崩 100×;pretrain 2×2 里 parabola h28 cos 0.95 时 nMSE=47659(数值发散)。**判决必须用 nMSE/pixel,cos 永不单用。**
2. **zero-shot 迁移天花板 = random 架构先验(0.607)**:训练只能恢复到接近、超不过;三条独立证据(物理方法全 <random / 增广不破 / epoch 越多越逼近)。出处:transfer_improvement_report。
3. **协议混淆可制造假阴性**:random-init × 单帧 probe × with-projector 三个 confound 乘性叠加,uniform vx 从假阴性 0.166 修到 0.939(paper-init + K=4 + no-projector);另有 init 静默丢 192 key 使 45-config sweep 全作废(⚠️ 6-2 sweep 数值不可引用)。出处:5-26 negtive_result_report、diagnostic_report。

## C7. 真实数据:zero-shot 死路 vs 直训活路 ✅✅

- zero-shot phyworld→Physion:封顶 0.607(random 先验),仅 Support(+0.10)/Collide(+0.07)有真信号。
- **Physion++ 直训**:刚体场景长程 cos 0.96–0.99,h64 仍 0.79;顶配 np20+scale h64 nMSE **0.087**(baseline 1/3.2)。
- 共同短板:布料形变(deform_clothhit 0.610 / clothhang 0.367)。
- 早期支撑:预训练域 > 规模 > task-FT(5.5M PushT ViT-tiny ≈ 749M ImageNet DiT-XL;vel_x 阶梯 pixel-stats 0.516→random 0.573→ImageNet 0.754→PushT 0.878→DiT 0.890);phyworld 上任何 SSL 微调 net-negative。

---

## 附:可信度警示

| 事项 | 状态 |
|---|---|
| 6-2 的 sweep_three_domains_results / piwm_three_domains_new | **作废**(init bug),不可引用 |
| 5-x 的早期 R² 数字 | 已被 5-26 协议修正推翻,只引修正后的 MSE+ρ |
| 多数 6-24/7-x 结果 | ⚠️ 单种子;P0 给 headline(C1)补 3 种子 |
| collision pixel 尺 | decoder-limited 不可信,collision 只用 latent 尺 |
| probe/structpos 2×2 的 pixel 尺 | 未完成(7-07 collector 被杀),P0 补 |
