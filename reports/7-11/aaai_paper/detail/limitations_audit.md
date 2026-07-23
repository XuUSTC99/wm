# Limitations 段审查(2026-07-23)

审查对象 = `paper/sections/5_conclusion.tex` 的 `\paragraph{Limitations.}`。结论:**有一条真遗漏、四点措辞瑕疵、两个高性价比可补实验**。所有判断都对着正文/附录原文核过,下面每条都带出处。

---

## 当前原文

> Our conclusions rest primarily on one compact backbone (∼10M ViT-tiny JEPA); the headline findings replicate on frozen DINOv2, where injection is largely inert rather than harmful—so the *harm* claim is established on the trainable-encoder regime. On the mechanism, the slot's use is demonstrated; the black box's role stays partly inferential—erasing its position copy takes 116–154 of 190 dimensions, at which rank amnesic projection loses selectivity (a limit we quantify, not a step we skipped), and wholesale replacement passes its controls in one of six cells.

覆盖了两类局限:① 单 backbone;② 机制归因里黑盒那半是推断的。

---

## 一、真问题:漏了两条本该在这儿的局限

### 1.1 ⚠️ 只测了"一个注入家族"——这是全文最大的适用边界,却没进 limitation

`3_method.tex` §3.2 明确写了两大类**不在**本文范围:

> Two families lie outside it and our results do not speak to them: designs that change what the representation *is* (object-centric slots, relational graphs) and designs that replace the predictor's functional form—the *transition operator* (Hamiltonian/symplectic integrators, learned contact).

conclusion 第二段也专门讨论了它:

> collision turns on contact and inter-object relation, which only operator-level designs (symplectic, learned contact) can express.

**问题**:这条边界在**正文出现两次**,却**没有出现在 limitation 清单里**。审稿人会觉得"你自己知道、也写了,却不肯放进 limitation"——这比没意识到更扣分。标题是 "Rethinking Physical Inductive Biases in Latent World Models"(泛指全部物理归纳偏置),实测只覆盖 intrinsic coordinate injection 一族,**标题与覆盖面之间的这个 gap 必须在 limitation 里显式认领**。

### 1.2 ⚠️ 域的规模/丰富度没提

三个 PhyWorld 域都是**单机制、ID 训练集仅 1000 条轨迹**(附录 §OOD partitions),Physion++ 用的是 readout split(800 clips)。"注入在更大数据/更丰富动力学下会不会翻盘"未被覆盖。§4.0 把小训练集说成是**受控设计的优点**(隔离数据量混淆),这没错,但**优点的另一面就是局限**,应当在此认领一句。

---

## 二、四点措辞瑕疵(不改数据,只改表述)

| # | 问题 | 依据 |
|---|---|---|
| **2.1** | **"passes its controls in one of six cells" 会被误读成"结论只有 1/6 支持"** | 附录原文是:*in five of six cells, removing position damages the rollout **less** than removing the same number of dimensions at random*。即**测试工具在这个表示上失去了选择性**,不是"结论弱"。当前措辞把"工具失效"读成了"证据不足",**自我贬低且不准确**。 |
| **2.2** | **两条 caveat 藏在附录、没上浮** | 附录自陈:① 控制**匹配 rank 但没匹配 variance**(INLP 选的是线性可预测方向,方差可能低于随机方向);② **早期跑到 24 维时曾出现"支持 bypass"的相反结果**,INLP 跑到收敛后消失。这两条恰恰是审稿人会问的,**主动写进 limitation 反而加分**(尤其②:它是对本文早期错误读法的自我纠正记录)。 |
| **2.3** | **"largely inert" 含糊** | 附录有确切数字:**23 of 27 cells 落在 baseline 种子带内**;明确变差的只有 dynamics-head 行(parabola 1.15×、collision 1.27×)与 label-free·collision(1.07×)。直接给数比 "largely" 强。另可一并说明**该 backbone 上注入方差约翻倍**(±0.087 vs baseline ±0.039),故单种子格在此 backbone 上不可用。 |
| **2.4** | ~~Physion++ probe 未测~~ | **已解决**(2026-07-23):probe/probe+slot ×3 种子已跑完,limitation 里该句已删,tab:pp 已加两列。此处仅作记录。 |

---

## 三、可补的实验(按性价比排)

### ⭐ P0:反向 counterfactual patching(patch 黑盒 → 读 slot)

**现状缺口**:Test 4 的 patching 是**单向**的——换 slot、看黑盒跟随(47–72%,对照 1.5–11%)。所以"slot 是 load-bearing"是**干预证据**;而"黑盒也 load-bearing"靠的是:

- Jacobian 占比 96–99%(**本质是维数多**,190 vs 2,说服力弱);
- 整体替换实验(**六格只有一格可用**,即 2.1 里那个问题)。

**做什么**:反过来——**把黑盒 190 维换成 donor 的,保持 slot 不动,测 slot/预测跟随 donor 多少**。

**为什么值**:
- 它**直接补掉 limitation 里 "the black box's role stays partly inferential" 这句**,把黑盒那半从推断升级为干预证据;
- 它**绕开 amnesic 失效**——不需要"擦除"(那正是 116–154 维那个死结),只需要"替换 + 读出";
- 与已有的正向 patching **完全对称**,协议现成、审稿人易懂。

**成本**:纯推理,在现有 ckpt 上跑,无需重训。3 域 × 3 种子 × 2 backbone,**约 1–2 小时**。

**预期与解读**(先说清楚,免得事后挑结论):
- 若黑盒跟随率**也高** → 两通道都 load-bearing,**冗余论证完整闭合**,limitation 那句可删;
- 若黑盒跟随率**低** → 说明预测其实主要读 slot,那**"两个通道都携带"这个说法要修**,如实改。两种结果都写。

### ⭐ P1:variance-matched 控制

**现状缺口**:附录自陈"控制匹配 rank 但未匹配 variance"。

**做什么**:除现有的 rank-matched 随机消融外,再加一组**方差匹配**的随机方向对照。

**为什么值**:把 2.2① 那条 caveat 从"我们承认有个弱点"变成"我们查过、结论不受影响"。成本同样是纯推理,**约 1 小时**。

### ❌ 不建议:operator-level 设计(Hamiltonian / symplectic / learned contact)

那是**另一篇论文的工作量**。正确做法是写进 future work + limitation 的适用边界(见 1.1),**不要**在 limitation 里暗示"我们本该做但没做"——那会把一个合理的范围界定,变成一个自认的缺失。

---

## 四、建议的执行顺序

1. **先跑两个实验**(P0 反向 patching + P1 variance-matched 控制),GPU 现已空闲,合计 2–3 小时;
2. 结果落地后**再改 limitation 文字**——因为 P0 的结果直接决定那句 "partly inferential" 是删掉、还是改写;
3. 文字改动一并处理 §2 的四点瑕疵。
