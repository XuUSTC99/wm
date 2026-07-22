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

四类,全部推理期、不重训。三域 × 三种子 × 两 backbone,数据已齐(累计约 210 个 run,
0 失败)。三类独立方法(steering / patch / Jacobian)在命题 (a) 上给出同一方向,这是
这条结论最强的地方。

| 方法 | 需要选"位置方向"吗 | 可用格数 | 对命题 |
|---|---|---|---|
| Steering(§3.1) | 是(slot 侧可直写) | slot 侧 5/6 | (a) 被推翻 |
| Counterfactual patch(§3.2) | 否(整体换 slot) | 3/6(阴性对照卡掉 3 格) | (a) 被推翻 |
| **Jacobian(§3.2b)** | **否** | **6/6** | (a) 被推翻 + 两路并存 |
| Amnesic(§3.3)/剂量阶梯 | amnesic 否、旧 patch 是 | 1/6 | (b) 部分 |

### 3.1 Steering —— 直接测命题 (a)

**一句话:偷偷把 slot 里存的位置挪一点,看模型自己算出来的位置跟不跟着挪。**

四步:

1. 拿一个训好的模型。latent 有 192 个数:前 2 个是注入的 slot(存位置),后 190 个是黑盒。
2. 在 rollout 过程中,把 **slot 里的位置往右挪 δ**。
3. 让模型继续预测下一步。
4. 去看**黑盒那 190 个数**现在说位置在哪。

判据:

- 黑盒也跟着挪了 → **模型读了 slot**(承重)
- 黑盒纹丝不动 → 模型没理 slot(不承重)

> **为什么看黑盒不看 slot**:slot 是我们刚写进去的,从它读回来当然是挪了的,说明不了
> 任何事。黑盒是**没碰过**的部分,它动了只能是因为 predictor 把 slot 的变化算进去了。
> 代码上还有一层保障:每步**先记录预测、再施加扰动**,所以读数永远取自模型自己算的东西。

增益 `g = 黑盒读数的变化 / δ`。`g=1` 表示完全承重,`g=0` 表示完全不承重。

阴性对照:往黑盒里推一个**等范数的随机方向**,看黑盒动不动 —— 用来排除"往 latent 里
加任何东西都会动"。

#### 怎么读这张表

**只看两个数,比大小:**

```
g_slot        推 slot 一格,黑盒动了多少
随机等范数对照  推同样大小的随机方向,黑盒动了多少
```

两个数差得远 → slot 被用了;差不多 → 没被用。

`±` 后面是三个种子的波动。**波动比数值还大就不能用**(见 parabola 那行)。

| backbone | 域 | g_slot | 随机等范数对照 | 差距 | 结论 |
|---|---|---|---|---|---|
| LeWM | uniform | **+1.518±0.033** | −0.017 | **90×** | ✅ slot 承重,三种子几乎一致 |
| LeWM | collision | **+2.005±0.204** | −0.003 | **600×** | ✅ slot 承重 |
| LeWM | parabola | +1.154 **±1.398** | −0.017 | — | ⚠️ **弃**:波动大于数值 |
| DINO | uniform | **+1.687±0.062** | −0.011 | **150×** | ✅ slot 承重 |
| DINO | parabola | **+0.586±0.300** | +0.001 | **590×** | ✅ 数值偏小但稳 |
| DINO | collision | **+1.944±0.308** | −0.031 | **60×** | ✅ slot 承重 |

**6 行里 5 行站得住,全部指向同一结论。**

举例读第一行:推 slot 一格,黑盒跟着动 1.5 格;推同样力气的随机方向,黑盒只动 0.017 格
(约等于没动)。差 90 倍 → 黑盒动不是因为"被推了一下",而是因为**模型真的读了 slot**。

被弃掉的 parabola 那行,三个种子分别是 **−0.097 / +0.897 / +2.663** —— 一个说完全不承重、
一个说强承重。**同一配置训出了三个本质不同的模型**(已验证不是 δ 尺度伪影,见 §5.3)。

→ **命题 (a) 被推翻。slot 是承重的。** 加性自检误差 <3%(说明干预落在线性区、没把
latent 推出流形),随机方向 ≈0。

> ⚠️ **那个 1.5 别当比例读** —— 不表示"承重 150%"。绝对值受探针 R²<1 的回归衰减、
> 动作输入未被扰动、3 帧历史锚定共同影响(§5.4)。只能用于"有 vs 没有"、
> "这条路 vs 那条路"的定性比较。

### 3.2 Counterfactual patch —— 直接测命题 (a)

**一句话:把 slot 里的位置整个换成另一条轨迹的,看模型跟着谁走。**

和 §3.1 的区别:steering 是"挪一点"(小扰动),patch 是**整个换掉**(换成另一条真实
轨迹的位置)。

四步:

1. 取两条轨迹:**本轨迹 A** 和 **供体轨迹 B**
2. rollout 时把 A 的 slot 换成 **B 同一时刻的位置**(黑盒 190 维不动,仍是 A 的)
3. 让模型继续预测
4. 看模型算出来的位置更像 **A** 还是更像 **B**

读数 **follow-fraction**:把预测出的位置同时对 A、B 的真值做回归,看 B 占多少权重。

```
follow = 0   完全跟着 A → slot 被无视
follow = 1   完全跟着 B → slot 说了算
```

#### 怎么读这张表

**左边那列是关键对照**:baseline 模型没做过注入,它的前 2 维只是任意两个数、没有物理
含义 —— 换掉它们,模型应该毫无反应。

| backbone | 域 | baseline(无 slot,阴性对照) | structpos(注入了) | 怎么看 |
|---|---|---|---|---|
| LeWM | uniform | **+0.015** | **+0.468** | 对照几乎不动 → 差别来自 slot |
| DINO | uniform | **+0.112** | **+0.650** | 同上 |
| DINO | collision | **+0.091** | **+0.721** | 同上 |

对照列 0.015–0.112(≈不跟随),注入后 **47–72%** 跟着供体走。

→ **差别不是"换东西会让模型乱",而是"换的是模型正在用的那个东西"。再次推翻命题 (a)。**

#### 为什么它和 steering 互补

| | steering(§3.1) | patch(§3.2) |
|---|---|---|
| 手法 | 挪一小点 | 整个换掉 |
| 强度 | 局部、线性区 | 大幅、用真实值 |
| 阴性对照 | 随机等范数方向 | 没有 slot 的模型 |

**手法不同、对照也不同,却给出同一结论** —— 这比任何单一方法可信。

#### 三格不可用

LeWM 的 parabola / collision:**阴性对照本身就脏** —— 那两维明明没有物理含义,baseline
却跟随了 0.42–0.51。原因**未查清**:试过共线性假说,parabola 说得通(pooled corr 0.756),
但 collision 只有 0.219 却同样脏。这两格弃用(见 §5.1)。

### 3.2b Jacobian —— 第三类独立方法,也指向 (a) 被推翻

**一句话:不动模型,直接问 latent 里哪一维对预测影响最大。**

前两个方法都要**动手改**中间状态(挪一点 / 换掉),这个方法**什么都不改** —— 纯求导。

三步:

1. 拿训好的模型,喂进一段历史
2. 对 latent 的**每一维**求导:「这一维动一点点,预测出的位置会变多少」
3. 得到 192 个敏感度数字,排序

看两件事:**① 每维强弱**(slot 每维敏感度 ÷ 黑盒每维敏感度)、**② 排名**(那 2 维在
192 维里排第几)。

因为不选方向、不设计干预,**不受**那个搞垮所有黑盒干预的"必须先选一个位置方向"的
问题影响 —— 36 格全部可用(剂量阶梯只剩 1 格)。

baseline 行是内建对照(未注入,dims[0:2] 只是任意两维):**6/6 的比值都在 0.80–0.98、
排名都在 90–162**,正是随机两维该有的样子。于是 structpos 行可信:

| backbone/域 | 比值(slot每维/黑盒每维) | slot 排名(共192) |
|---|---|---|
| LeWM uniform | **3.78±0.31** | **1, 2** |
| LeWM parabola | **4.00±0.39** | **2, 1** |
| DINO uniform | **2.29±0.37** | **1, 7** |
| LeWM collision | 1.76±0.35 | 9, 59 |
| DINO collision | 1.27±0.56 | 33, 125 |
| DINO parabola | 0.99±0.11 ❌ | 111, 90 |

**注入把那 2 维从第 99 名推到第 1–2 名。** 同时 slot 只占**总敏感度的 0.8–4.0%**
(仅 2 维 vs 190 维)→ 这是"两条路都承重、slot 强在效率黑盒强在体量"的解析证据。

#### 这是不是常用方法(引用已查证,2026-07-22)

**是,而且是可解释性里最早的一族**,但有两条成熟批评,恰好决定了我们该怎么用它。

梯度敏感度的谱系:

- **Saliency map** —— Simonyan, Vedaldi & Zisserman, *Deep Inside Convolutional Networks:
  Visualising Image Classification Models and Saliency Maps*, arXiv:1312.6034 (2013)。
  对输入求类别分数的梯度。
- **Integrated Gradients** —— Sundararajan, Taly & Yan, *Axiomatic Attribution for Deep
  Networks*, ICML 2017, arXiv:1703.01365。沿基线到输入的路径积分梯度,满足 sensitivity 与
  implementation invariance 两条公理。
- 我们做的是对**中间表示**求导(不是输入像素),在动力学/世界模型里分析学到的转移算子时属常规操作。

两条必须知道的批评:

1. **梯度 ≠ 因果效应。** 梯度是局部线性的 —— 说的是"变一丁点会怎样",不是"真改掉会怎样"。
   这正是 amnesic probing 那一系(§3.3)存在的理由:Elazar, Ravfogel, Jacovi & Goldberg,
   *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals*, TACL 2020,
   arXiv:2006.00995 —— 其核心主张就是 **"conventional probing performance is not correlated
   to task importance"**,要求对"从探针读数推因果"保持怀疑。
2. **部分显著性方法通不过健全性检验。** Adebayo, Gilmer, Muelly, Goodfellow, Hardt & Kim,
   *Sanity Checks for Saliency Maps*, NeurIPS 2018, arXiv:1810.03292。其 Model Parameter
   Randomization Test 把模型权重逐层随机化,发现某些方法的输出几乎不变 —— 即它反映的可能
   是输入结构而非模型学到的东西。

**→ 所以本研究的用法是站得住的:梯度与干预并用,且结论一致。** 两类方法的弱点不重叠 ——
梯度的标准批评是"不因果",有 steering/patch 顶上;干预的标准批评是"你的干预设计有偏"
(我们确实吃过这个亏,最小范数干预废掉 5/6 格),有 Jacobian 顶上。

另外,§3.2b 的 baseline 对照(未注入时 slot 两维排第 99/90 名,正是随机两维该有的位置)
**直接回应了第 2 条批评** —— 我们的读数对"模型是否被注入过"是敏感的,不是输入结构的产物。

#### 限制

Jacobian 是**局部线性**敏感度,只测单步预测对最后一帧历史的响应,不等同于 rollout
全程因果效应;与干预法互补,不可互相替代。

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

#### ⚠️ 一个必须一起写的混淆因素

**INLP 挑的方向与随机挑的方向,统计性质不同。** INLP 专挑"线性可预测位置"的方向,这些
可能是**低方差**方向;随机挑则会打到高方差方向。所以"删位置伤害更小"**可能来自选择偏差,
而非位置真的不重要**。

即:这个对照只做到了**秩匹配**,没做到**方差匹配**。

写作时必须主动交代 —— 它**进一步支持**"该测试不具判别力"这个结论,而不是削弱它。
(若要补,可加一个"方差匹配的随机对照":按 INLP 所删方向的方差谱去采样随机方向。
成本低,但目前没做。)

#### 📌 这部分该怎么写进论文

**结论:能写,而且应该写。** 三个理由:

1. **审稿人会点名要 amnesic**(它是该领域标准做法,Elazar et al., TACL 2020)。与其被问
   "为什么不做",不如主动交代"做了,并量化了它为何在此表示上不能判别"。
2. **负面结果里包着一个正面数字** —— 116–154/190 是全文"冗余、分布式"最硬的量化,
   远强于现稿的"随机 2 维对照失败"。这个数字**独立于 amnesic 判不判得出来**,该进正文。
3. **失败的原因恰恰印证主张** —— 不是执行失误,是位置铺得太开,任何能抹掉它的投影同时
   抹掉了别的一切。**"做不了"是冗余程度的推论**,自洽。

不要写成"我们失败了",写成"我们跑了标准测试并刻画了它的判别边界":

> We ran amnesic projection with a rank-matched random-ablation control. Erasing position
> requires removing 116–154 of the 190 non-slot dimensions; at that rank, removing position
> is indistinguishable from removing the same number of dimensions at random (in five of six
> cells it is in fact *less* damaging). The test cannot discriminate in this representation —
> a direct consequence of how distributed the copy is. We note the control is matched in rank
> but not in variance: INLP selects directions that linearly predict position, which may be
> lower-variance than randomly chosen ones.

**放置建议:**

| 位置 | 写什么 |
|---|---|
| §4.3 机制节 | 116–154/190 这个冗余量化(正面结果) |
| Traps 附录 | amnesic 不具判别力 + 迭代不到底的陷阱 + 方差不匹配的坑 |

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
| **(a) slot 不承重** | ❌ **被推翻**(三类独立方法一致) | steering g_slot 1.5–2.0(随机方向≈0);patch 跟随 47–72%(无 slot 对照 1.5–11%);Jacobian slot 排名 1–2/192(baseline 同两维排 ~99) |
| **(b) 黑盒承重** | ⚠️ **部分** | 剂量阶梯在 LeWM/uniform 证实(整体替换黑盒 → 跟随 0.958),但该测法 6 格只有 1 格通过对照;amnesic 这条路无选择性,走不通 |
| 两条路**并存**(非二选一) | ✅ 已证 | 钉 slot 不置换黑盒副本(Test 1);Jacobian 显示黑盒仍占总敏感度 96–99% |

### → "bypass" 这个词现在不配用

因为 (a) 被推翻。正确的机制句是:

> 注入并没有被绕过。往 slot 写一个位置偏移,模型自己的黑盒状态跟着移动 1.5–2.0 倍
> (随机等范数方向 ≈0);把 slot 换成另一条轨迹的,预测跟随供体 47–72%(无 slot 的
> 对照只跟随 1.5–11%);而 Jacobian 显示注入把那 2 维推到了全 192 维敏感度的第 1–2 名
> (baseline 的同两维排在 ~99)。但黑盒里那份冗余副本一点没被挤掉,而且靠 190 维的
> 体量仍占总敏感度的 96–99%。
>
> **两条路都在承重 —— slot 每维最强,黑盒总量占优 —— 而预测误差纹丝不动。**
> 这才是"再注入同一份信息不带来新信号"的因果版本。

这个改动**保住了框架的大部分**(冗余存在、未被置换、加权无效、约束生效四条全在),
只修正"slot 被忽略"这一个过头推论。而且更难反驳:审稿人不能再说"你没让它承重"
或"你注入得不够狠"。

---

## 5. 还缺什么

### 5.1 结构性缺口(影响结论)

| 缺口 | 严重度 | 说明 |
|---|---|---|
| **命题 (b) 只覆盖一个域** | 高 | 剂量阶梯(整体替换 k 维黑盒)是有效测法,但 6 格里只有 LeWM/uniform 通过对照门。amnesic 那条路无选择性、原理性走不通。措辞只能停在"两路都承重",不能说黑盒是主路 |
| **parabola 在三种方法上都异常** | 中高 | LeWM steering 三种子不稳(已证非 δ 伪影)、dose ladder 对照脏、DINO Jacobian 唯一失败格(比值 0.99)。同一个域反复出问题,原因未查清 |
| **collision 的 slot 从未真正吸收位置** | 中 | slot R² 仅 0.40–0.43,Jacobian 比值也最弱(1.27–1.76)。该域"slot 是否承重"这个问题本身就问得不干净 |
| **patch 阴性对照脏** | 中 | LeWM 的 parabola/collision baseline 也跟随 0.42–0.51,而 baseline 的 dims[0:2] 无物理含义 → 这两格有非特异效应,不能单独用。查过共线性假说:parabola 的 pooled corr 0.756 说得通,但 collision 只有 0.219 却同样脏 → **原因未查清** |
| **绝对增益不可解释** | 低(需声明) | g>1 受 probe R²<1 的回归衰减、未扰动动作输入、3 帧历史锚定共同影响;structdyn 上更证实 g 会随 δ 与构造方式变动一个量级以上。只能定性比较 |

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
