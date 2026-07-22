# "load-bearing" 的定义、已做实验、以及还缺什么

2026-07-22。这份文档回答三个问题:论文里"可承重"到底怎么定义的、我们做了哪些实验、
每个实验证到了哪一步、还差什么。

相关:[causal_bypass_results.md](causal_bypass_results.md)(干预数据)、
[08_causal_bypass_plan.md](../08_causal_bypass_plan.md)(方案与坑)

---

## 1. 论文里的定义

### 1.1 核心区分:decodable ≠ load-bearing

摘要原句:

> the physical state is **decodable, but not load-bearing**

两个词的含义:

| 术语 | 含义 | 怎么测 |
|---|---|---|
| **decodable**(可解码) | 线性 probe 能从表示里读出位置 | ridge 回归,报 ρ 或 R² |
| **load-bearing**(可承重) | **预测在 rollout 时真的用了这一路** | 必须干预:动这一路,看预测变不变 |

论文自己援引的标准是 amnesic probing \citep{elazar2021amnesic}:

> probe-recoverable information need not be information the model uses

**所以"可承重"是一个因果/使用性概念,不是信息存在性概念。** 光证明"位置在里面"
不构成承重,必须证明"预测读了它"。

### 1.2 正式假设(§3.3)

> **Hypothesis (the load-bearing problem).** The physical slot occupies 2 of 192 dimensions
> and receives ~1% of the prediction gradient, while the black-box channel *redundantly*
> encodes the same position — **so prediction can route around any injected slot.**

拆成两个必须分别成立的命题(方案文档 §1 也强调了这点):

- **(a) slot 不承重**:干掉/篡改 slot,rollout 的位置几乎不动。
- **(b) 黑盒承重**:干掉黑盒里的位置子空间(slot 完好),rollout 的位置塌。

> ⚠️ 只有 (a) → 可能"两路都没在用"(机制叙述要整个换)。
> 只有 (b) → 可能"两路都在用",那就不是 *bypass* 而是 *shared routing*,措辞必须改。
> **"bypass"这个词只有 (a)∧(b) 同时成立才配用。**

---

## 2. 论文当前做了哪些实验

### 2.1 §3.3 里声明的三个测试

| 测试 | 做法 | 结果 | **证到了什么** |
|---|---|---|---|
| **Test 1**<br>约束注进去了吗 | 分别从全 192 / 黑盒[2:190] / slot[0:2] / 随机 2 维解码位置 | 黑盒 ρ=0.92/0.85/0.79 ≈ 全 192;随机 2 维 0.2–0.5 失败 | 冗余副本**存在**,一条不经过 slot 的路**可用** |
| **Test 2**<br>约束真生效吗 | 加权物理损失 / 预测损失比;encoder 参与率(PR) | 比值 15–125×;PR 塌 39–90% | 约束**真在起作用**(排除"没接上"的 bug) |
| **Test 3**<br>加权能救吗 | pos_weight 1→300 扫描 | 只回到 parity,从不净赢 | 失效**不是注入强度不够** |

### 2.2 外部对照

| 实验 | 结果 | 证到了什么 |
|---|---|---|
| PIWM 移植(extrinsic) | ID 强,但 encoder-OOD 塌(ρ 0.33 vs 0.89) | 架构上堵死旁路**确实**能让物理状态承重 → 从外部支持机制的架构预言 |
| 跨 backbone(冻结 DINOv2) | 黑盒 ρ=0.951 ≥ 全 latent;LeWM 0.899 | 冗余与旁路是**通用视觉表示的属性**,不是我们训练的产物 |

### 2.3 关键缺口(论文自己写明了)

§4.3 结尾:

> Together they make one account by far the most economical — prediction keeps reading the
> copy the black box already carries — **but we do not intervene on a trained model, so that
> last step is inference.**

也就是说:**上面五个实验没有一个包含"predictor 在读哪一路"这个变量。**
Test 1 只证了"路存在",Test 2 证了"约束生效",Test 3 证了"不是强度问题"。
从"存在一条路"到"predictor 走那条路",现稿是用**排除法**跨过去的。

§4.3 结尾那段也诚实标注了 Test 1 的两个限制:probe 读的是 **frame embeddings 而非
rolled-out latents**,而且 **linear decodability need not imply use**。

---

## 3. 新做的因果干预(尚未进论文)

三种,全部推理期干预、不重训。三域 × 三种子 × 两 backbone,数据已齐。

### 3.1 Steering —— 直接测命题 (a)

往一路写位置偏移 δ,看模型**自己预测出的**黑盒状态跟着动多少。`g = Δ读数 / δ`。

| backbone | 域 | g_slot | 随机等范数对照 | 结论 |
|---|---|---|---|---|
| LeWM | uniform | **+1.518±0.033** | −0.017 | slot **承重** |
| LeWM | collision | **+2.005±0.204** | −0.003 | slot **承重** |
| LeWM | parabola | +1.154±1.398 ⚠️ | −0.017 | 不稳,不可用 |
| DINO | uniform | **+1.687±0.062** | −0.011 | slot **承重** |
| DINO | parabola | **+0.586±0.300** | +0.001 | slot **承重** |
| DINO | collision | **+1.944±0.308** | −0.031 | slot **承重** |

→ **命题 (a) 被推翻。slot 是承重的。** 加性自检误差 <3%,随机方向 ≈0。

### 3.2 Counterfactual patch —— 直接测命题 (a)

把 slot 换成另一条轨迹同时刻的 slot,看预测跟随供体多少(follow-fraction)。

| backbone | 域 | baseline(无 slot,阴性对照) | structpos |
|---|---|---|---|
| LeWM | uniform | **+0.015** | **+0.468** |
| DINO | uniform | **+0.112** | **+0.650** |
| DINO | collision | **+0.091** | **+0.721** |

→ 没有 slot 就不跟随,有 slot 就跟随 47–72%。**再次推翻命题 (a)。**
(LeWM 的 parabola/collision 阴性对照本身就有 0.42–0.51 跟随,那两格不可用,见 §5。)

### 3.3 Amnesic —— 本想测命题 (b),但做不了

把位置子空间从黑盒 190 维里投影掉。

| 域 | 臂 | 干净 | 删黑盒位置 | **删同秩随机(对照)** |
|---|---|---|---|---|
| uniform | baseline | 0.115 | 0.737 | **1.030** |
| uniform | structpos | 0.146 | 0.626 | **1.099** |
| parabola | baseline | 0.347 | 0.887 | **1.031** |
| parabola | structpos | 0.372 | 0.844 | **1.021** |
| collision | baseline | 0.337 | 1.028 | **1.128** |
| collision | structpos | 0.402 | 1.161 | 1.097 |

**两个独立结论:**

**(i) 冗余的最硬量化** —— 要删掉 190 维里的 **116–154 维(60–80%)** 才能把位置线性抹掉。
远强于现稿的"随机 2 维对照失败"。**这个数字该进论文。**

**(ii) 但这个实验丧失了选择性** —— 6 格里 5 格"删位置"的伤害**小于**"删同秩随机"。
那个"崩"不是因为删了位置,而是因为删掉了 145+ 维。
→ **命题 (b) 用 amnesic 测不出来。这条路原理性走不通。**

> 📌 陷阱记录:早期只跑 12 次迭代(24 维)时显示"删位置比删随机伤害小",看着支持旁路;
> 跑到收敛后对比消失。典型的"干预没做到底就下结论"。

### 3.4 弱信号:有 slot 确实缓冲了黑盒的损失

按各自干净基准归一化的退化倍数:

| 域 | 无 slot(baseline) | 有效 slot(structpos) |
|---|---|---|
| uniform | 6.4× | **4.3×** |
| parabola | 2.6× | **2.3×** |
| collision | 3.05× | **2.9×** |

三域一致:有 slot 的模型退化更少。方向符合"slot 能接管",但幅度小、且被选择性问题淹没,
**不能单独当证据**。

---

## 4. 结论:每个命题现在处在什么状态

| 命题 | 状态 | 依据 |
|---|---|---|
| 冗余副本**存在**于黑盒 | ✅ 已证 | Test 1;amnesic 需删 116–154/190 维加强 |
| 钉 slot **不置换**黑盒副本 | ✅ 已证 | Test 1 第三行;structpos 的 g_bb 不低于 baseline 满量程 |
| 约束**真生效** | ✅ 已证 | Test 2(损失比 15–125×、PR 塌 39–90×);干预侧 g_slot 1.5–2.0 |
| 失效**不是强度问题** | ✅ 已证 | Test 3 加权到 300 只回 parity |
| **(a) slot 不承重** | ❌ **被推翻** | steering g_slot=1.5–2.0;patch follow 47–72% |
| **(b) 黑盒承重** | ⚠️ **未证** | amnesic 无选择性,测不出来 |

### → "bypass" 这个词现在不配用

因为 (a) 被推翻。正确的机制句是:

> 注入并没有被绕过。往 slot 写一个位置偏移,模型自己的黑盒状态跟着移动 1.5–2.0 倍
> (随机等范数方向 ≈0);把 slot 换成另一条轨迹的,预测跟随供体 47–72%。但黑盒里那份
> 冗余副本一点没被挤掉。**两条路都在承重,而预测误差纹丝不动。**
> 这才是"再注入同一份信息不带来新信号"的因果版本。

这个改动**保住了框架的大部分**(冗余存在、未被置换、加权无效、约束生效四条全在),
只修正"slot 被忽略"这一个过头推论。而且更难反驳:审稿人不能再说"你没让它承重"
或"你注入得不够狠"。

---

## 5. 还缺什么

### 5.1 结构性缺口(影响结论)

| 缺口 | 严重度 | 说明 |
|---|---|---|
| **命题 (b) 无法证明** | 高 | amnesic 是唯一想到的直接测法,但在这个表示上无选择性。若始终测不出,措辞只能停在"两路都承重",不能说黑盒是主路 |
| **LeWM·parabola steering 不稳** | 中 | 三种子 −0.097/+0.897/+2.663,两个坐标都不稳。同域 DINO 稳定为正 → 是 LeWM-parabola 特有问题,原因未查(可能 δ 尺度或伪逆条件数) |
| **patch 阴性对照脏** | 中 | LeWM 的 parabola/collision baseline 也跟随 0.42–0.51,而 baseline 的 dims[0:2] 无物理含义 → 这两格有非特异效应,不能单独用 |

### 5.2 已知但不打算补的

| 项 | 为什么不补 |
|---|---|
| 恢复实验的第 4 臂("删黑盒 + 打乱 slot") | 该设计的臂 2 前提就不成立(删位置 ≯ 删随机),补第 4 臂救不回来 |
| routing index `R = g_bb/(g_bb+g_slot)` | 对冗余通道系统性失效:baseline 里黑盒是**唯一**通路,g_bb 也只有 0.02–0.33,说明它是方法噪声地板而非因果估计 |

### 5.3 三项后续 —— 已全部跑完(2026-07-22 晚,27 run,0 失败)

细节见 [causal_bypass_results.md §6](causal_bypass_results.md);
复现 `python phyworld/scripts/collect_causal_followup.py`。

| 项 | 结果 | 对论文 |
|---|---|---|
| **Test 1 改用 rolled-out latents** | ✅ **成立**。黑盒 0.841(LeWM)/0.880(DINO)≈ 全 latent 0.866/0.911,随机 2 维 0.471/0.231 失败。而且这 42 个 run 早就跑完了,只是没人读 | **删掉现稿那句自我限制**("the probe here reads frame embeddings, not rolled-out latents"),白赚 |
| **structdyn 正对照** | ❌ 定量不可用 / ✅ 定性可用。换构造方式没用,δ 变小反而更大(4.5→17.6),collision 还会随构造反号(−228 vs +417)。但 g_bb 全部 ≈0 → R→0 方向正确 | g 的绝对值**必须声明不可解释**;R→0 可作定性正对照 |
| **LeWM·parabola 不稳原因** | ❌ **是真实差异,不是 δ 伪影**。δ 缩小 20 倍后逐种子值纹丝不动(−0.056 vs −0.097),相对离散度稳定在 1.10;uniform 同阶梯只有 0.02–0.19 | 写成 limitation:该域该 backbone 三个种子学出了**本质不同的模型**(slot 承重 −0.06 / +1.10 / +2.61),不要平均掉 |

> 📌 方法论收获:B2(uniform 同 δ 阶梯)是专门加的对照。没有它,parabola 的 δ 无关性
> 无法与"方法本身到处都有 δ 依赖"区分开 —— 结论就立不住。

### 5.4 绝对增益不可解释(写作时必须声明)

`g` 超过 1(uniform 1.5、collision 2.0)受 probe R²<1 的回归衰减、未扰动的动作输入、
3 帧历史锚定共同影响。**只能作定性比较,不能当"承重比例"读。**

---

## 6. 对论文的具体改动清单

**必改:**

- `0_abstract.tex:2,5` — "decodable, but not load-bearing" / "the injected one is never load-bearing"
- `1_introduction.tex:19` — "leaves the injected slot bearing none of the load" / "The state is decodable, but not load-bearing"
- `1_introduction.tex:26` — contribution 2 的 "decodable but not load-bearing"
- `3_method.tex:64` — Hypothesis 里 "prediction can route around any injected slot"
- `4_experiments.tex:64,68,124` — 图 caption 与 §4.3 正文
- `5_conclusion.tex:4` — 同名论断

**可删的安全网:**

- `4_experiments.tex:68` 结尾 "we do not intervene on a trained model, so that last step is inference"
  —— 现在做了干预,但结论与原预期相反,要换成干预的实际结果。

**该新增的:**

- amnesic 的 116–154/190 数字(冗余最硬量化)
- amnesic 无选择性 → 审稿人点名的这条路做不通(负面结果,进 rebuttal 有力)
- baseline 的 slot R² 只有 0.04–0.19(干净的阴性对照)
