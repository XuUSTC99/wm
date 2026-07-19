# 2026-07-17 论文修改记录：术语统一 + 摘要重写 + Introduction 对齐

本次修改共 5 项，全部位于 `paper/sections/`，每次改动后均重新编译两遍，`main.log` 零错误，`main.pdf` 已更新（最终 20:41）。

---

## 1. 全文术语：`photorealistic` → `realistic`

**动机**：Physion / Physion++ 原论文的自我表述是 "realistic simulation"，与其对齐。

共替换 14 处，涉及 7 个文件：

| 文件 | 处数 | 代表位置 |
|---|---|---|
| `0_abstract.tex` | 1 | "two realistic-simulation benchmarks" |
| `1_introduction.tex` | 2 | anatomy 段 + contributions 第一条 |
| `2_related_work.tex` | 1 | "realistic---not camera-captured---simulation" |
| `3_method.tex` | 1 | "realistic scenes"（horizon 匹配段） |
| `4_experiments.tex` | 5 | benchmark 介绍、Table 2 caption、Figure 6 caption、"realistic scale/dynamics" |
| `5_conclusion.tex` | 2 | 结论首段 + limitations "realistic *simulation*, not camera video" |
| `A_appendix.tex` | 1 | augmentation 边界段 |

注意：related work 与 conclusion 中原有的 "not camera-captured / not camera video" 澄清语**刻意保留**——改为 realistic 后这两处澄清更有必要。

**勘误（同日 20:47）**：最初的全局替换在 4 处产生了错误搭配——`photorealistic` 原本修饰的不是 simulation，直接换成 realistic 后（"realistic scenes/scale/dynamics"）会被误读为真实世界数据。已改为 `realistic-simulation` 复合修饰语：
- `3_method.tex`："collision; realistic-simulation scenes"（horizon 匹配段）
- `A_appendix.tex`："keeps helping at realistic-simulation scale"
- `4_experiments.tex`："realistic-simulation scenes have the largest appetite"
- `4_experiments.tex`（Figure 6 caption）："realistic-simulation dynamics reward long-rollout training"

其余 10 处 realistic 均直接修饰 simulation（"realistic-simulation benchmarks"、"realistic---not camera-captured---simulation" 等），无歧义，维持不变。

## 2. 摘要整体替换（`0_abstract.tex`）

按用户提供的新文本整体重写。新旧叙事差异：

- **新增开篇定义句**：world model = "predicts how an environment evolves in response to actions by rolling an internal representation forward"。
- **puzzle 框架**：无物理监督 → 位置可解码、单步近精确 → 多步 rollout 漂离自己编码的动力学。
- **轴命名改变**：旧 "state/evolution targets" → 新 "**physical quantities and physical laws**"；"five mechanism families" 不再出现在摘要（正文保留）。
- **机制表述**："decodable but not load-bearing" → "injected copy is **superfluous**；prediction reads the state from the internal representation and **bypasses** the injected one"。
- **结尾新主张**："injected structure **should** supply what the representation is missing---*how the state evolves, not what the state is*"。
  - 措辞演变：初版 "must" → 讨论后定为 "should"（用户改定）；"would have to" 曾作为备选。
  - **风险与对策**：论文中钉 evolution 的 Family 3/4/5 同样失败，此句可能被审稿人反问；对策见第 3 项（conclusion 承认 + future work）。
- **删除的旧内容**（仍存在于 intro/conclusion，仅摘要不提）：frozen-DINOv2 跨 backbone 复现；extrinsic 架构 "necessary but not sufficient"；具体 ρ 数值区间；re-weighting 2/4 parity。

LaTeX 化细节：em-dash 用 `---`；数字进 math mode（`$29$ of $30$`、`$2.2$--$8.3\times$`）；*correct* 与结尾句用 `\emph{}`；各句间仅换行不空行（保持 AAAI 单段摘要）。

## 3. Conclusion 承认 evolution 方法的局限（`5_conclusion.tex`）

**动机**：为摘要结尾 "should supply how the state evolves" 提供正文落点，堵住 "你们自己的 evolution 注入不也失败了吗" 的反问。

两处新增：

1. **"constructive reading" 段末新增**：我们的三个 evolution 注入变体（kinematic heads、consistency loss、label-free prior）是为覆盖设计轴而刻意选择的简单实现（"deliberately simple instantiations chosen to span the design axes"）；它们的失败表明钉 evolution 不是自动有效的（not *automatically* useful），但**并未穷尽**更强的 dynamics-matched evolution 注入的设计空间——设计这样的机制是 future work。
2. **Limitations 段末扩写**：velocity-on-parabola 那句后补 "our evolution-targeting mechanisms are simple instantiations that may understate what a better-designed dynamics injection could deliver"。

**让步边界**：只承认"我们的实现简单、可能低估上限"；不动摇针对 state 注入的机制结论（bypass / load-bearing），也不承诺 evolution 注入一定可行。

## 4. Introduction 对齐新摘要（`1_introduction.tex`）

**第 1 段（开篇）**：
- 换用摘要定义句式（"predicts … by rolling an internal representation forward … expected to respect physical regularities"）。
- 改为 puzzle 叙事结构；保留全部原数字（ρ≤0.96、cosine 0.98–0.99 → 0.24、Figure 1 引用）。
- 段尾由 "State decodability does not imply physical-law compliance" 改为 "The model fails not by lacking the physical state, but by failing to *evolve* it"，为结尾 evolution 论点埋线。

**第 3 段**："inject the physics into the model" → "inject **physical structure** into the latent representation"；核心问题句 "can injecting **physical structure** make a latent world model physically consistent?"。

**第 5 段（anatomy）**：轴命名对齐摘要并加桥接注释——"hard and soft constraints on both **physical quantities (what the state is) and physical laws (how it evolves)**, with labeled and label-free supervision"（括号解释使读者可对上 Table 1 的 state/evolution 表头）。

**第 6 段（结果）**：
- 机制句改为摘要口径："already encoded across the internal representation … an injected copy is superfluous … reads the state from the representation and bypasses the injected one"。
- "even the exact gravitational form a=g hurts" → "even supplying the *correct* physical constraint a=g hurts"。
- 嵌入摘要关键句："**What the model lacks is not the state but the ability to evolve it over long horizons**"，引出 protocol 2.2–8.3×。

**未改动**：Figure 1 caption、related-work 对比段（第 2 段）、prior-evaluations 段（第 4 段）、contributions 列表。

## 5. 拼写统一：`labelled` → `labeled`

摘要用美式 "labeled"，正文原为英式 "labelled"。统一为美式（AAAI 为美国会议）：`3_method.tex` 6 处（含 Table 1 各 family 表头与 caption）、`4_experiments.tex` 1 处。

## 6. §3.3 新增 free rollout 成因分析（`3_method.tex`，20:57）

在 exposure bias 句之后、horizon 句之前插入一段解读（用户待润色）：free rollout 的增益不只是对累积误差的鲁棒性——监督 $H$ 步自回归 rollout 使训练本身成为**长程动力学模拟**（long-horizon dynamics simulation）：teacher forcing 只考察局部单步转移，free rollout 考察整个 horizon 上的状态演化，梯度奖励自洽的多步动力学而非单步映射——正是 latent 缺失的能力（呼应 §1）。由此 free rollout 是一种**无需物理标签的 evolution-targeted 训练信号**：训练 "how the state evolves" 而不触碰 "what the state is"（呼应摘要/结论的 constructive 落点）。

## 7. 全文去掉 "post-hoc"，改用 fine-tune 口径（2026-07-19）

**动机**："post-hoc fine-tuning" 语义冗余（fine-tuning 本就发生在预训练之后），且 post-hoc 真正修饰的是注入时机而非训练动作；直接用 "fine-tuning a pretrained encoder vs. from-scratch co-training" 自解释、更简单。

共 7 处：
- `3_method.tex`：regime 句改为 "fine-tuning a pretrained encoder vs.\ from-scratch co-training; with injection on/off this gives a $2\times2$ per domain"（顺带把 2×2 的两个因子说明白）。
- `4_experiments.tex`：正文 "(post-hoc)" → "(fine-tuned)"；Table 3 行标 "post-hoc" → "fine-tune"（3 行）；Table 3 caption "Pretraining vs.\ post-hoc injection" → "Injection under from-scratch co-training vs.\ fine-tuning a pretrained encoder"。
- `1_introduction.tex`：anatomy 段 "(post-hoc fine-tuning and …)" → "(fine-tuning a pretrained encoder and …)"；结果段 "from-scratch vs.\ post-hoc" → "from-scratch vs.\ fine-tuned"。

事实口径备注：post-hoc/fine-tune regime = encoder 从 PushT checkpoint 初始化后在物理域带注入 loss 继续训练（"后训练嫁接"），不是从收敛的 PhyWorld baseline 出发。

## 8. 删除原 Figure 3（bypass 柱状图，2026-07-19）

删除 `fig15_bypass_probe190.pdf` 的 figure 环境及 `Figure~\ref{fig:bypass}` 引用（正文数字 ρ=0.79–0.92、随机对照 0.2–0.5 已完整覆盖图中信息）。后续图自动重编号：fig:lbr→3、fig:fr→4、fig:ladder→5、fig:dinowmscan→6。无悬空引用。

## 9. §3.2/§4.3 Test 2 简化 + 数据勘误（2026-07-19）

- §3.2 测试 2 与 §4.3 Test 2 段简化（去掉 (a)/(b) 编号、≈1/≫1 符号条件、"not a measured gradient norm" 长 hedge，保留 "proxy for gradient magnitude"）。
- **勘误**：§4.3 原句 "at λ=1 the ratio is benign yet the harm persists" 与数据不符——diagnostic_report.md §1.2（fixed-init，λ∈{1,5,10}）实测 λ=1 时 (λ·probe)/pred = **3–20×**、probe 占 total 20–43%（"pred 项已被相对边缘化"），并非 benign。已改为 "the ratio is milder ($3$--$20\times$) yet the harm persists"。修正后逻辑也更自洽：λ=1 下约束同样在实质用力，危害与剂量连续。

---

## 验证

- 每步 `pdflatex` × 2，`grep "^!" main.log` 均为空（零错误）。
- `grep -rn photorealistic sections/` 为空；`grep -c labelled sections/*.tex` 全 0。
- 最终 `main.pdf` 382,710 bytes，2026-07-17 20:41。

## 遗留 / 待讨论

- contributions 第一条仍用旧轴命名口径（未出现 state/evolution 字样，暂不冲突）；如需改为 "physical quantities / physical laws" 口径可再改。
- 摘要不再提 frozen-DINOv2 复现与 extrinsic "necessary but not sufficient"——若想保留跨模型卖点，可在摘要倒数第二句后补半句。
- 摘要结尾 "should" 句的防御依赖第 3 项的 conclusion 让步；若审稿人仍攻此句，备选措辞 "would have to"（虚拟语气）在讨论中已给出。
