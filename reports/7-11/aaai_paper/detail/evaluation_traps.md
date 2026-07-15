# 论据:判决指标怎么选 —— cos / probe-ρ / nMSE / pixel 的优劣与比较

> # 🎯 一句话结论
> **cos/probe 是"训练目标的对偶量"——加了对应 loss 必然涨、不代表预测变好(数学必然);拿它们当主指标会系统性**高估**物理结构。实锤有二:①我们实测多处 cos/probe 升而 nMSE/pixel 反转(下方三案例);②我们自己早期 sweep 踩过坑——盯 K=4 probe-ρ 得"λ=50 胜出",改用可信指标 pred_loss 后翻案(λ=1 最弱 probe 反而最优)。**但也没有单一完美指标:nMSE 自己有除零爆点(方向与 cos 陷阱相反)。**故判决 = **nMSE + pixel 为主、cos/probe 只当诊断、逐分区逐 horizon 交叉验证**。(不宣称"多少文献如此"。**注:deep-sup 2504.03861 本身用的就是可信指标 pred_loss、结论正确,不是本陷阱的例子;其 recipe 搬到我们高维视觉 latent 上失效属塌方/稀释机制,见 [why_physics_structure_fails.md](why_physics_structure_fails.md)。**)**

**对应主张**:[01_results_ledger.md](../01_results_ledger.md) **C6** / [06_storyline.md](../06_storyline.md) 发现三
**配图**:[../figures/fig5_cos_trap.png](../figures/fig5_cos_trap.png)

![](../figures/fig5_cos_trap.png)

**怎么读这张图(cos 陷阱一眼版)**:三个真实案例,每个两根柱——**蓝柱=cos 指标怎么说,红柱=真正重要的指标(pixel/nMSE)怎么说**。纵轴是"好坏比值"(相对 baseline,**>1 更好、<1 更差**,虚线 1.0=baseline,log 轴)。

- **每个案例都是:蓝柱在 1.0 之上(cos 说"变好了")、红柱在 1.0 之下(真值说"变差了")** → 这就是陷阱:**只看 cos 会得出和事实相反的结论**。
- 三个案例(越往右陷阱越狠):① **probe** cos 1.05(↑)但 pixel 0.97(↓);② **app 增广 h64** cos 1.10(↑)但 nMSE 0.90(↓);③ **app 增广 deform** cos 1.50(↑ 大涨)但 nMSE **0.21**(↓ 崩成 1/5)。
- 之所以能背离:cos 只量**方向**、不量**尺度**——预测幅度过冲/崩塌它看不见(详见 §0 判据②);而 nMSE/pixel 带尺度、量"预测对不对"。**所以判决必须用 nMSE/pixel,cos 只当诊断。**
- 出处:probe 案例(**uniform 域 h28**)[probe_vs_structpos_summary.md](../../6-24/probe_vs_structpos_summary.md)(§2.2 cos 表 / §3.2 pixel 表,probe 列);增广两案例 `/data1/.../runs/physionpp/eval_pp_fr{,_app05}_e20.log`(deform_clothhit:cos 0.610→0.913、nMSE 0.772→3.69)。

---

## 0. 四把尺子:各自量什么、优劣在哪

**先定义清楚**(详见 [02 名词表·四把尺子](../02_story_and_novelty.md)):

| 尺子 | 量的是什么 | 优 | 劣 |
|---|---|---|---|
| **cos** | 预测 latent 与真实未来 latent 的**方向**余弦(1=同向) | 尺度无关、数值稳、不会被退化分母引爆 | **尺度盲**——幅度过冲/崩塌看不见;且是 structured/probe loss 的**对偶量**(加 loss 必涨) |
| **probe-ρ** | 另训线性头从 latent 读出物理量,读出值 vs 真值的 Pearson 相关 | 直接量"信息**在不在**"(可读出性),是 presence 的唯一直接测量 | 只量存在、不量预测对不对;**被 probe loss 直接优化 = 自己给自己打分**;且受 **range restriction** 影响、分区间不可直接比 |
| **nMSE** | `‖预测−真值‖²/真值方差`,即**差向量的长度** | 展开 `‖p−t‖²=‖p‖²+‖t‖²−2‖p‖‖t‖cosθ` **显式含 cosθ** → **方向偏或尺度偏都罚**;对物理注入是外部指标 | **分母是真值方差 → 方差→0 时除零引爆**(见 §2) |
| **pixel PSNR** | 预测 latent 解码成图 vs 真实帧的逐像素质量 | **最强的锚**:端到端、可人眼验证、**最难作弊**(改 latent 分布骗不过逐像素比较,除非真预测对了球的位置) | 贵(要解码);受重建质量上限拖累 |

**一句话对比**:**上两个量"存在/方向",下两个量"预测对不对"**。

**先讲清楚:没有单一完美指标——nMSE 自己也有坑(除零爆点,见 §2)。我们不是"信 nMSE 不信 cos",而是一个能当"判决"的指标要满足两个判据,cos/probe 违反、nMSE/pixel 满足:**

**判据①:独立于被检验的干预(不循环)。**
- **probe-ρ 循环**:用 probe loss 训练,probe-ρ 就是被优化的那个量,加了 probe 必然涨——拿它评 probe 方法 = **自己给自己打分**;structured slot 可解码性同理。
- **nMSE/pixel 不循环**:物理结构方法优化的是额外的 structured/probe loss,不是 rollout nMSE、更不是 pixel;对"加不加物理结构"这个干预,它们是**外部**指标。

**判据②:直接对齐"预测对不对"这个真实目标。**
- 世界模型目标 = 预测未来状态准。nMSE 直接量"预测 latent 与真实未来 latent 差多少"(带尺度);pixel 量"解码画面 vs 真实画面"。
- cos 只量**方向**(尺度盲);probe-ρ 只量"信息**在不在**"——都不直接量"预测**对不对**"。

**为什么信 nMSE**:它的可信度部分来自"**和更难作弊的 pixel 同向**"——所有反转案例里,分歧的都是 cos vs (nMSE/pixel),**nMSE 与 pixel 始终一致**。

**所以判决规则不是"只信 nMSE"**,而是:nMSE + pixel 为主(量"对不对")、cos/probe 为诊断(量"方向/存在")、**逐分区逐 horizon 交叉验证**,分歧时查清原因(cos 尺度盲?nMSE 分母退化?)——任何单一指标都不单独下结论。

---

## 1. 陷阱一:cos 陷阱(cos 升而真值崩)

cos/K4-ρ 是 probe/structured loss 的对偶量,加对应 loss 必然涨,不代表预测变好。实锤反转(goodness 比值,>1 更好):

| 案例 | cos 说 | 真值指标说 |
|---|---|---|
| probe（uniform, latent cos vs pixel） | h28 cos 0.882 vs 0.843（**升**） | pixel PSNR 19.93 vs 20.64 dB（**降**） |
| app-aug h64（cos vs nMSE） | 0.794→0.870（**升**） | nMSE 0.280→0.311（**退化**） |
| **app-aug deform_clothhit** | 0.610→0.913（**升 +0.30**） | nMSE 0.772→3.69（**崩**） |

**判决必须用 nMSE/pixel,cos 永不单用。** 源:probe [probe_vs_structpos_summary.md](../../6-24/probe_vs_structpos_summary.md)(uniform,§2.2/§3.2 probe 列);真实 `/data1/.../runs/physionpp/eval_pp_fr{,_app05}_e20.log`。

## 2. 陷阱二:nMSE 自身的除零爆点(与陷阱一方向相反)

parabola h28 附近个别轨迹球出框 → 目标 latent 方差→0 → **nMSE 除零飙 3 万~197 万**,而同 horizon 的 cos 仍 0.55~0.95 正常。both-OOD 聚合被这几条拉爆(六个 parabola 臂全中招)。**规则:引 nMSE 前先查 by-horizon 是否发散;parabola 判决走 r/m-OOD。** 源:[NOTE_from_lewm_pretrain_caveat.md](../NOTE_from_lewm_pretrain_caveat.md)。

**两个陷阱方向相反,正是"必须交叉验证"的理由**:cos 尺度盲会**漏报**(真值崩了它看不见);nMSE 分母退化会**虚报**(预测没那么差它却爆表)。任何一把尺子单用都会翻车。

---

## 论文表述

> **cos 无尺度会漏报,nMSE 有尺度会被退化分母引爆——必须逐分区、逐 horizon 双指标交叉验证;pixel 是最难作弊的锚。** 附:训练 loss 收敛 ≠ rollout 泛化好(scratch 120ep pred_loss 收敛到与 pusht 同量级 0.008,rollout r/m-OOD 仍差 2.8×)。

**落点**:给社区的评测 checklist——**判决用 nMSE/pixel、cos/probe 只当诊断、逐 horizon 查除零爆点、probe-ρ 不跨分区比(range restriction)**。

---

*本文档只管"指标本身怎么选"。另两条与指标无关的方法论坑各有归宿:**zero-shot 迁移天花板 = random 架构先验(0.607)** → [real_data_physion.md](real_data_physion.md);**协议混淆制造假阴性**(random-init × 单帧 probe × with-projector,vx 0.166→0.939)→ [01_results_ledger.md](../01_results_ledger.md) C6 段。*
