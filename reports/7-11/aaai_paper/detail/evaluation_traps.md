# 论据:四个评测陷阱(方法论贡献)

> # 🎯 一句话结论
> **既有文献"看起来"物理结构有效,大半是用了 cos/probe 这类"训练目标的对偶量"(加了对应 loss 必然涨,不代表预测变好);我们给出四个系统性评测陷阱 + 一份修正协议,回收"别人为什么被骗"。**

**对应主张**:[01_results_ledger.md](../01_results_ledger.md) **C6** / [06_storyline.md](../06_storyline.md) 发现三
**配图**:[../figures/fig5_cos_trap.png](../figures/fig5_cos_trap.png)（陷阱1）、[../figures/fig6_transfer_ceiling.png](../figures/fig6_transfer_ceiling.png)（陷阱2）

![](../figures/fig5_cos_trap.png)

---

## 陷阱 1:cos 陷阱(cos 升而真值崩)

cos/K4-ρ 是 probe/structured loss 的对偶量,加对应 loss 必然涨,不代表预测变好。实锤反转（goodness 比值,>1 更好）:

| 案例 | cos 说 | 真值指标说 |
|---|---|---|
| probe（latent cos vs pixel） | h28 cos 0.882 vs 0.843（**升**） | pixel PSNR 19.93 vs 20.64 dB（**降**） |
| app-aug h64（cos vs nMSE） | 0.794→0.870（**升**） | nMSE 0.280→0.311（**退化**） |
| **app-aug deform_clothhit** | 0.610→0.913（**升 +0.30**） | nMSE 0.772→3.69（**崩**） |

**判决必须用 nMSE/pixel,cos 永不单用。** 源:probe [kinematics_exploration.md](../../6-24/kinematics_exploration.md);真实 `/data1/.../runs/physionpp/eval_pp_fr{,_app05}_e20.log`。

## 陷阱 2:zero-shot 迁移天花板 = random 架构先验(0.607)

训练只能恢复到接近、超不过。三条独立证据:① 所有物理方法 <random（pos_weight 0.551 最差）;② 增广不破（0.597）;③ epoch 越多越逼近（1→20:0.554→0.603）。**别把"接近 random"当成"学到了迁移能力"。** 源:[transfer_improvement_report.md](../../physion/transfer_improvement_report.md)、`reports/physion/eval_*.json`。

## 陷阱 3:协议混淆制造假阴性

random-init × 单帧 probe × with-projector 三个 confound 乘性叠加,uniform vx 从假阴性 0.166 修到 0.939（paper-init + K=4 + no-projector）。另有 init 静默丢 192 key 使 45-config sweep 全作废（⚠️ 6-2 sweep 数值不可引用）。源:5-26 negtive_result_report、diagnostic_report。

## 陷阱 4:nMSE 自身的除零爆点(与陷阱1方向相反)

parabola h28 附近个别轨迹球出框 → 目标 latent 方差→0 → nMSE 除零飙 3 万~197 万,而同 horizon 的 cos 仍 0.55~0.95 正常。both-OOD 聚合被这几条拉爆（六个 parabola 臂全中招）。**规则:引 nMSE 前先查 by-horizon 是否发散;parabola 判决走 r/m-OOD。** 源:[NOTE_from_lewm_pretrain_caveat.md](../NOTE_from_lewm_pretrain_caveat.md)、[parabola-bothood-nmse-blowup memory]。

## 论文表述(两个方向合起来)

> **cos 无尺度会漏报,nMSE 有尺度会被退化分母引爆——必须逐分区、逐 horizon 双指标交叉验证。** 附:训练 loss 收敛 ≠ rollout 泛化好（scratch 120ep pred_loss 收敛到与 pusht 同量级 0.008,rollout r/m-OOD 仍差 2.8×）。

**落点**:给社区的评测 checklist（判决用 nMSE/pixel、逐 horizon 查爆点、probe 协议三件套、迁移看 random 天花板）。
