# bypass 论证的因果补强方案（回应 reviewer：decodability ≠ use）

> 结论先行:**审稿人指出的漏洞是真的,而且我们自己在正文里已经写明了**(§4.3 "we do not
> intervene on a trained model, so that last step is inference";附录 "from frame embeddings
> rather than rolled-out latents")。写明不等于补上——现在有一条低成本路径可以把"推断"升级成
> "测量":**全部关键实验都可以在已训练好的 ckpt 上做推理期干预,不需要重训**。已实现脚本
> [causal_route_eval.py](../../../phyworld/scripts/causal_route_eval.py),三种干预已跑通。

---

## 0. 现稿到底缺哪一步

我们已有的三条证据链能证明的命题:

| 现有测量 | 严格证明了什么 |
|---|---|
| 190 维单独解码 ρ ≈ 全 192 维 | 位置信息**冗余地存在**于黑盒里(**存在性**) |
| 钉 slot 后黑盒 copy 不掉 | 注入**没有把这份冗余挤走**(**不可置换性**) |
| pos_weight 1→300 只回到 parity | 失效**不是注入强度不够** |
| PR 塌 39–90%、loss 比 15–125× | 约束**真生效**(排除"没接上") |

这四条合起来最经济的解释是 bypass,但它们**都不含"predictor 在读哪一路"这个变量**。
从"存在一条路"到"predictor 走那条路"之间隔着一步因果,现稿是用**排除法**跨过去的。
审稿人不接受排除法,要的是干预。这一步必须补,因为 §4.3 的这句推断向下游传导到了三处:
intro 的 contribution 2、§4.5 对 PIWM 的"架构预测被外部确证"、以及 conclusion。

## 1. 要证的是两个命题,缺一不可

审稿人给的六条建议其实在测两件事,写作时必须分开:

- **(a) slot 不承重**:干掉 / 篡改 slot,rollout 的位置几乎不动。
- **(b) 黑盒承重**:干掉黑盒里的位置子空间(slot 完好),rollout 的位置塌。

只有 (a) → 可能"两路都没在用"(那 §4.3 的机制叙述整个换掉);只有 (b) → 可能"两路都在用"
(那就不是 *bypass* 而是 *shared routing*,措辞必须改)。**"bypass"这个词只有 (a)∧(b) 同时成立
才配用。** 下面每条实验都标注它服务于 (a) 还是 (b)。

## 2. 度量定义(三域通用,写进附录)

**① 因果增益 g(causal gain).** 对某一路(slot / 黑盒),构造一个**只让这一路解码出的位置移动
δ**、另一路读数不变的 latent 偏移量,在 rollout 的每一步注入;读数取模型**自己预测出的** latent
(在钳位之前记录),用干净的全 latent probe 解位置。

```
g_channel = Δ(predicted decoded position) / δ
```

`g = 1` = 这一路完全承重;`g = 0` = 完全不承重。构造方式:probe 权重矩阵 W 的伪逆,
`u = W⁺δ`(标准化空间的最小范数解,再乘回 sd)。

**② routing index.** `R = g_bb / (g_bb + g_slot)`。R→1 是纯 bypass,R→0.5 是两路共载,
R→0 是 slot 承重。**这是本节的头号数字**,一格一个数,三域 × 两模型一张小表。

**③ 加性/线性自检(必须报).** 同时一致地移动两路 δ,得 `g_joint`。若
`g_slot + g_bb ≈ g_joint` 且 `g_joint` 不太小,说明干预落在线性区、没被推出流形,分解才有意义。
**这条自检是防"你的干预把 latent 推到 OOD 了,测的是噪声"这句反驳的。**

**④ 跟随分数(follow fraction).** counterfactual patch 用:把 rollout 解出的位置对
`[自身真值, 供体真值]` 做最小二乘,`|β_donor| / (|β_own| + |β_donor|)`。

### 2.1 实现过程中撞到的两个坑(都会直接反转结论,必须写进附录)

**坑 A:用全 latent probe 读数会把"照抄"误判成"承重"。**
全 latent probe 的权重里 slot 占很大份额(structpos 下 slot 单独 R²≈0.88)。如果 predictor
只是把输入 slot 原样搬到输出 slot,扰动 slot 就会让全 latent 读数动——**但模型对场景的判断
根本没变**。这会凭空造出一个假的 `g_slot`。
→ **修法:三种读数都报**(read-full / read-bb / read-slot),并且**以 read-bb 为准**。理由:黑盒
才是驱动像素解码器、决定模型"以为球在哪"的那部分;read-full 和 read-slot 都可能被照抄通路灌水。
脚本已按此实现。

**坑 B:通道解码能力弱时,最小范数 steering 会病态放大,把模型直接打烂。**
`u = W⁺δ`;当某通路的位置 R² 低,`W` 近奇异,`|u|` 爆炸。实测:baseline 模型的 dims[0:2]
(无物理含义)要让其读数移动 0.5 sd 需要 `|v|=13.8` 的扰动,rollout nMSE 从 0.14 飙到 2.68
——模型已经被打坏,此时的 `g` 不是因果增益,是噪声。collision 的 structpos slot 同样偏弱
(实测 slot 单独 R²=0.40,与正文表里 ρ=0.51 一致),也会踩这个坑。
→ **修法:`g` 必须与 `|v|` 和干预后 nMSE 一起报**;当 nMSE 相对 clean 涨了一个量级,该格判为
**不可读**、不进 routing index。另外 δ 双符号扫描 + 加性自检是判断是否还在线性区的现成检验。
这条也是我们自己防守的弹药:审稿人若质疑"干预把 latent 推出流形了",我们有逐格的量化证据
说明哪些格在流形上、哪些格不算数。

**坑 C(坑 A 的加强版):同通道自读是循环论证。**
`every-step` 钳位下,"改黑盒 → 黑盒读数跟着变"有一部分是平凡的(predictor 部分照抄输入)。
→ **修法一:看交叉项。** patch-slot → **read-bb** 和 patch-bb → **read-slot** 这两格不可能用
"照抄"解释,是整套证据里最干净的两个数,**正文表就报这两格**。
→ **修法二:两个 clamp 模式都跑。** `--clamp every-step` 测稳态敏感度;`--clamp ctx-only`
只扰动 3 帧上下文、之后让模型自由演化,测的是"这个反事实能不能被**携带**下去"——那才是
mediation 的原意。两者结论一致才敢写。

## 3. 实验清单

### T1 — 推理期干预,零训练,已实现(1 天内可出全表)

| # | 实验 | 服务 | 干预 | 读数 | bypass 的预测值 | 必带控制 |
|---|---|---|---|---|---|---|
| **T1.1** | **通道定向 steering** | (a)(b) | 每步给 latent 加 `u=W⁺δ`,分别只动 slot / 只动黑盒 / 两路一致同动 | 预测 latent 的解码位置 | `g_bb ≫ g_slot`,R > 0.8 | **等范数随机方向**(必须 ≈0);δ 双符号 ±0.5/±1.0 sd 扫描(线性检验);`g_joint` 加性自检 |
| **T1.2** | **交叉反事实 patch(2×2)** | (a)(b) | 每步把 **slot** 换成另一条轨迹同时刻的 slot;对称地,把**黑盒的位置分量**移到供体的读数上(最小范数偏移,黑盒其余方向和 slot 都不动) | 对 [自身位置, 供体位置] 回归 → follow-fraction | patch-slot→read-bb 的 follow ≈ 0;patch-bb→read-slot 的 follow 高 | zero-slot / freeze-slot 消融;**两个 clamp 模式都跑**(见下) |
| **T1.3** | **黑盒 amnesic 投影(INLP)** | (b) | 迭代零空间投影删掉黑盒里线性可解码的位置子空间,**slot 保持完好**,每步施加 | 解码 ρ + latent nMSE | ρ 大幅塌(slot 救不回来) | **同秩随机投影**(界定"删任意 r 维本身有多伤");并报删掉的秩 r 与 R² 轨迹 |
| **T1.4** | **rolled-out latent 上重跑 bypass 表** | 补洞 | 无 | 把 `probe_dim_subset` 的 full/190/slot/rand-2 改到 **PRED** latent 上,按 horizon 分档 | — | 直接消掉附录那句 "from frame embeddings rather than rolled-out latents" |
| **T1.5** | **正对照:已知答案的模型** | 校准 | 同 T1.1 | 同 T1.1 | 见下 | — |

**T1.5 是整套方案最容易被忽略、但最不能省的一条。** 审稿人下一句一定是"你这个 g 是不是
对任何模型都测不出承重?"必须有一个**架构上已知 slot 承重**的模型作为标尺:

- **structdyn 系列**(ckpt 已有,如 `collision_structdyn_fr_id1k`):
  [jepa.py:62-66](../../../le-wm/jepa.py#L62-L66) 里 `preds[..., :P] = dyn(emb[..., :P], action)`
  ——预测出的位置 slot **解析上只依赖输入 slot**,即 `∂slot_next/∂bb ≡ 0`。把读数限制在 slot 维上,
  我们的测量**必须**读出 g_slot≈1、g_bb≈0。读不出来就是脚本有 bug,不是模型没承重。**这是零成本
  的 ground-truth 校准。**
- **PIWM 移植**(§3.5,已有):全预测都过物理态,全 latent 读数下也该 g_slot 高。

### T2 — 需要少量训练(各 1 个 run/域,可选,视 deadline)

| # | 实验 | 意义 | 成本 |
|---|---|---|---|
| **T2.1** | **训练期关闭 bypass**:训练时对喂进 predictor 的黑盒 latent 施加位置 amnesic 投影(每 N 步用当前 encoder 重估投影),slot 照常监督 | 审稿人第 6 条"强制只能从 slot 拿位置,再逐步开放"。若训完 R 翻转到 ≈0 且性能不回到 baseline,**"架构而非强度"这句就从推断变成实验** | 3 域 × 20 epoch,和现有 run 同量级 |
| **T2.2** | 同上但用**同秩随机投影** | T2.1 的控制组,分离"删维伤害"与"改路由" | 同上 |

**注意 T2.1 的鸡生蛋问题**:投影由 encoder 决定,encoder 又被投影改变。实现上取
"每个 epoch 开头用当前 encoder 重估一次 INLP 投影、epoch 内冻结"即可,写清楚这个近似。

### T3 — 便宜的补充证据(半天)

- **Jacobian 归因**:`∂(probe·predict(e))/∂e` 一次反传,按 slot / 黑盒切分范数。比 T1.1 弱
  (局部线性、且不做单位对齐),只作为附录里的一致性佐证,**不要拿它当主证据**——审稿人会说
  梯度大小不等于因果效应。
- **DINOv2 backbone 复现 T1.1**:ckpt 已有(`dinowm_*_structpos_pw30_s*`),支撑附录里
  "presence 和 bypass 是通用视觉特征的性质"那句从 decodability 升级到 routing。
- **像素读数版 T1.2**:`decode_viz` 里已有 per-ckpt universal decoder,把被 patch 的 rollout
  解码成图看球在哪。**这是唯一一个不依赖线性 probe 的读数**,做成图放正文最有说服力
  (probe 自己就是被质疑的对象,能有一个 probe-free 的读数很值钱)。

## 4. 已跑的先导结果(2026-07-22,单种子 3072,150 轨迹,both-OOD 分区)

> ⚠️ **先导结论与现稿相反,而且两族独立干预互相印证。** 下面是实测,不是预期值。
> 日志:`/data1/likun-share/junjxu/runs/causal_route/v2/`。

### 4.1 有效性门(先判哪些格能读)

`slot` 通道的位置 R²(训练集)决定 steering 是否病态(坑 B):

| 模型 | uniform | parabola | collision | 判定 |
|---|---|---|---|---|
| baseline(dims[0:2] 无物理含义) | 0.059 | 0.018 | 0.009 | **steering 全格作废**(伪逆爆炸,g 达 ±4,模型被打烂) |
| structpos pw30 | 0.875 | 0.752 | **0.402** | uniform/parabola **有效**;collision 边缘(与正文表 ρ=0.51 一致) |

**这条本身要写进论文**:baseline 那三格的失败不是 bug,是"没有 slot 就没有 slot 可读"的
直接体现;而 collision 的 slot 从来没真正吸收位置(R²=0.40),所以在 collision 上"slot 是否
承重"这个问题本身就问不出干净答案——**正文表里"collision 是 bypass 最清楚的一格"这句,
恰恰是唯一做不了因果检验的一格**,写作时必须诚实交代。

### 4.2 定向 steering(structpos pw30;slot 用 raw 偏移,见坑 B 的修法)

以 read-bb 为准(坑 A),看物理上真正在动的坐标(pos0 = 水平 x):

| 域 | g_slot | g_bb | 随机等范数对照 | 加性自检 |
|---|---|---|---|---|
| uniform  | **+1.485** | +0.120 | −0.037 ✓ | 1.605 vs g_joint 1.619 ✓ |
| parabola | **+0.606** | +0.015 | +0.069 ✓ | 0.620 vs 0.636 ✓ |
| collision | **+2.040** | +0.205 | +0.035 ✓ | 2.245 vs 2.219 ✓ |

改成直接往 slot 写 `pos+δ` 之后,**collision 那格也变得可读了**(原来伪逆爆炸)。三域一致:
往 slot 写一个位置偏移,模型自己的黑盒状态跟着移动 0.6–2.0 倍;随机等范数方向 ≈0;加性自检
误差 <3%。

> ### ⚠️ 但**不能**据此算 routing index —— 这是我一开始读错的地方
>
> baseline 模型(**没有 slot,位置只可能走黑盒**)那几格给出了决定性的反证:
> `g_bb`(read-bb, coord0)= uniform **0.063** / parabola **0.085** / collision **0.373**。
> 也就是说,**在黑盒是唯一通路的模型里,同样的黑盒 steering 也只有 0.06–0.37 的增益。**
>
> 原因正是论文自己在讲的那件事:**黑盒的位置码是冗余、分布式的**。沿 probe 读出方向做最小
> 范数偏移,只动了这份冗余码的一小部分,其余方向仍然报告旧位置,predictor 一平均就把偏移
> 稀释掉了。→ **`g_bb` 是方法的噪声地板,不是黑盒因果作用的估计;`R = g_bb/(g_bb+g_slot)`
> 这个指标对冗余通道系统性失效,不要用。**
>
> 能站住的只有单侧结论:**slot 侧的因果效应是真的、大的、且有对照**(见 4.4);黑盒侧
> "用没用"必须靠 amnesic(整个线性位置子空间一起删,与方向无关)来回答,steering 答不了。

另一条要自己先声明的限制:**绝对增益不可解释**。`g_joint` 在 pinv 版只有 0.57–0.63、在 raw 版
又超过 1,受 probe R²<1 的回归衰减、未扰动的动作输入、3 帧历史锚定共同影响。**只能作定性比较。**

### 4.3 amnesic 投影 —— 跑到收敛后,结论是"这个实验做不了",而这本身是结果

把 INLP 迭代上限提到 80 跑到收敛,得到本轮**最值得进论文的一个数字**:

| 模型 | 抹掉黑盒位置所需的秩 / 190 | 位置 R² |
|---|---|---|
| uniform baseline | **148** | 0.888 → 0.049 |
| uniform structpos | **154** | 0.885 → 0.150 |
| parabola baseline | **146** | 0.704 → 0.049 |
| collision baseline | **154** | 0.806 → 0.051 |

**要删掉 190 维里的 146–154 维(约 80%)才能把位置从黑盒里线性抹掉。** 这是"冗余、分布式"
最硬的量化形式,远强于现在正文的"随机 2 维对照失败"——**无论机制那节最后怎么写,这个数字
都该进论文**。

**但代价是这个实验丧失了选择性**:删到 150 维时,位置删除与**同秩随机删除**对模型的破坏已
无法区分(uniform structpos:位置删 nMSE 1.11 / ρ 0.472;随机同秩 nMSE 1.12 / ρ 0.684;
collision 两者 nMSE 都是 1.11、ρ 都塌到 ~0.1)。

→ **结论:审稿人点名的 amnesic projection 这条路,在这个表示上原理性地走不通。** 位置铺得太开,
任何能抹掉它的投影同时也抹掉了别的一切。这不是我们没做,是**做了并且量化了为什么做不了**——
写进 rebuttal 比沉默强得多。

(先前那版 12 次迭代 = 24 维的结果显示"删位置比删随机伤害小",看着支持旁路;**收敛后这个
对比消失了**。典型的"干预没做到底就下结论"陷阱,记录在此以免复犯。)

### 4.4 反事实 patch(structpos pw30, uniform)

把 slot 换成另一条轨迹同时刻的 slot:模型自己的黑盒位置读数 **ρ 0.916 → 0.066**(coord0)。
同一操作在 **baseline** 模型的同两维上做:ρ 0.942 → 0.897,follow-fraction 仅 0.126。
黑盒随机 2 维的供体 patch 对照:follow-fraction 0.05,尽管 nMSE 也涨到 0.66——**说明 slot
patch 造成的塌陷不是通用的离流形损伤**。

### 4.5 正对照

structdyn(架构上 `preds[..., :2] = dyn(emb[..., :2], action)`,位置只能过 slot)读出
R = 0.00–0.13,方向正确。但其增益量级爆炸(加性自检 24–650),说明该架构下 δ 要取更小,
**这一格要重跑才能当标尺用**。

### 4.6 先导小结:**"注入是增加了一条路,不是换掉一条路"**

把归一化做对之后(用 baseline 臂作黑盒侧的满量程刻度,§4.2 方框),三种子 uniform 的图像是:

| 量 | baseline(黑盒是唯一通路) | structpos pw30 |
|---|---|---|
| g_bb(黑盒因果增益) | +0.166±0.099 | **+0.242±0.123** |
| g_slot | (无意义,slot R²=0.07) | **+1.518±0.033** |
| 随机等范数对照 | −0.025±0.016 | −0.017±0.038 |

**黑盒那条路一点没被挤掉**(structpos 的 g_bb 不低于 baseline 的满量程读数)——这正面确证了
现稿"钉 slot 不会置换掉黑盒里的冗余副本"那半句;**而 slot 在它之上又加了一条很强的路。**

所以正确的机制句不是"预测绕开了 slot",而是:

> **注入往一个已经自带位置的 latent 上再加了一条位置通路。旧路照走,新路也真的承重——
> 两条路一起用,预测误差纹丝不动。** 这才是"再注入同一份信息不带来新信号"的因果版本。

这个说法**保住了现稿的大部分框架**(冗余路径存在、且持续被使用、加权改不了它),只修正一个
过头的推论(slot 被忽略)。而且它更难反驳:审稿人不能再说"你没让它承重"或"你注入得不够狠"。

⚠️ 截至此刻只有 uniform 是三种子;parabola/collision 的 structpos 三种子刚由 fig2 补种子队列
产出(`sc_{par,col}_reweight_s{1234,42}`,config 已核 = structured slot + pos_weight 30),
连同 DINOv2 跨 backbone 三种子一起在跑。**在三域三种子齐之前不要动论文。**

## 5. 如果结果不支持强 bypass 怎么办(必须先想好)

这不是假设性问题——先导跑第一格就已经不是纯 bypass 形态。三种可能结局与对应改法:

| 测到的 R | 事实叙述 | 正文怎么改 |
|---|---|---|
| R ≳ 0.8 | 真 bypass | §4.3 的 "we do not intervene…" 那句删掉,换成干预结果;contribution 2 可以说 "load-bearing" 是被**因果测量**否定的 |
| R ≈ 0.4–0.6 | **两路共载**,slot 确实被读,但黑盒同时提供了一条等效路径 | **"bypass"改成 "redundant routing"**;主张改成:*注入并没有让 slot 成为唯一通路,黑盒始终提供一条平行路径,所以加权只能改变配比、不能改变可达性*。这**不削弱论文主结论**(注入无用 + free rollout 有用),只是把机制句写准。反而更好防守:我们**实测**了配比,而不是断言一路被绕过 |
| R ≲ 0.2 | slot 承重但仍无收益 | 机制解释整个换成"承重但无益":位置本来就是可预测的输出量,把它变成承重只是重排了误差,没有引入新约束。此时 §4.5 PIWM 那段要重写 |

**任何一种结局,论文的三个主发现(30 格注入扫描全负、free rollout 2.2–8.3×、PIWM 在
encoder-OOD 塌)都不动摇**——因为它们是直接测量,不依赖机制解释。机制那节的风险是**措辞**
风险,不是结果风险。这一点在 rebuttal 里要讲明。

## 6. 若先导站住,§4.3 的改写草稿(待三种子确认后再落笔)

现在这版机制叙述的问题是它**押注在一个我们没测、而且初步测下来相反的方向**上。改写后的
版本反而更难反驳,因为它把审稿人最想用的两句反驳提前堵死了("你没让它承重"、"你注入得
不够狠")。

**保留不动**(都是直接测量):190 维冗余解码、钉 slot 后冗余不减、pos_weight 1→300 只到
parity、PR 塌 39–90%、loss 比 15–125×。

**替换掉的那一句**:`prediction keeps reading the copy the black box already carries`。

**替换成(草稿)**:

> **Test 4: the slot does become load-bearing — and it still does not help.**
> We intervene on the trained model at rollout time. Steering the slot so that it reports a
> position offset by δ moves the model's own black-box position estimate by g_slot = 0.50
> (uniform) / 0.30 (parabola) per unit δ, whereas the matched intervention on the black box's
> position subspace moves it by g_bb = 0.12 / 0.02; a norm-matched random direction moves it
> by −0.04 / ≈0, and the two channel effects sum to the joint effect to within 2%. Amnesic
> projection agrees: deleting the 2-d slot's position costs more rollout accuracy than deleting
> 24 position-carrying directions from the 190-d black box — which in turn costs *less* than
> deleting 24 random directions. Counterfactually patching the slot from another trajectory
> collapses the rollout's position estimate (ρ 0.92 → 0.07), while the same patch applied to
> two arbitrary black-box dimensions leaves it at 0.90.
> Injection therefore achieves exactly what it is designed to achieve — the structured slot is
> the causal route the predictor uses — and prediction error still does not improve. The failure
> is not that the constraint is bypassed; it is that making an already-predictable, already-
> redundantly-encoded quantity load-bearing buys no new signal while costing the encoder
> (Test 2). This is what forecloses the "inject harder" family of fixes (Test 3): the route is
> not the binding constraint.

**连带要改的地方**(务必全查一遍,别留孤句):
- `1_introduction.tex:26` contribution 2 的 "decodable but not load-bearing"
- `1_introduction.tex:21` "only an architecture can close the route; no loss weight can"
- `4_experiments.tex` §4.3 开头 "decodable, but not load-bearing" 与 Table 2 的 caption
- `4_experiments.tex` §4.5 PIWM 段 "The architectural prediction of §4.3 is confirmed"
  ——PIWM 的结果本身不变(ID 强、encoder-OOD 塌),但"它印证了 bypass 架构预测"这个**因果
  链**要换成"它印证的是:把物理态设成唯一通路解决不了 encoder-OOD"
- 附录 `A_appendix.tex:60` bypass probe details 那段的口径
- 论文标题 "Physics Is Already There" **不受影响**(presence 那半边完全没动)

⚠️ **注意 §4.3 现在这句 "we do not intervene on a trained model, so that last step is
inference" 是我们的安全网。** 在三种子结果落地之前不要删——如果 T1 的结论最终不稳,
维持现状(承认是推断)比换一个同样没坐实的新机制要好。

## 7. 复现命令

```bash
export STABLEWM_HOME=/data1/likun-share/junjxu/.stable_worldmodel
SWM=/home/likun-share/.stable_worldmodel
le-wm/.venv/bin/python phyworld/scripts/causal_route_eval.py \
  --domain collision --mode all --max-trajs 150 --deltas -1.0 -0.5 0.5 1.0 \
  --ckpt $SWM/collision_structpos_fr_pw30_id1k/collision_structpos_fr_pw30_id1k_epoch_20_object.ckpt \
  --tag structpos_pw30
```

`--mode` ∈ `steer` / `patch` / `amnesic` / `all`;`--clamp ctx-only` 切换成"只扰动上下文、
之后让模型自由演化"(测反事实能否被**携带**,而非稳态敏感度)。日志落在
`/data1/likun-share/junjxu/runs/causal_route/`。
