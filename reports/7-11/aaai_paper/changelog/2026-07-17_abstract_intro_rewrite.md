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

## 10. Introduction 按 AAAI 六段式蓝图重写（2026-07-19）

按外部意见把 intro 从 7 段重构为六段式，并对齐新 abstract 的机制框架（redundant→bypassed / what-the-state-is vs how-it-evolves）。

- **第 1 段（谜）**：加快开场（去掉慢热的通用定义句），保留全部数字（ρ≤0.96、cos 0.98–0.99、collision h28→0.24、OOD several-fold），收在 "state is present; fails to evolve it"。
- **第 2 段（药方+缺口）**：合并旧 line13（remedy/data-scaling）与旧 line17（factorial gap）为一段；显式点出 "shared, high-dimensional latent" 与 "training protocol 混淆" 双重缺口。
- **第 3 段（做了什么）**：**删除全部 setup 细节**——JEPA 定义、TF/FR 定义、PhyWorld [0.7,1.5] 参数范围与四分区、nMSE 定义全部搬走（归 Setup §4.1）；压成"十变体 × 两 regime × 三域 + 两真实感基准 + 指标纪律一句话"。"exhaustive"→"systematic"。
- **第 4 段（结论一：注入失败）**：判定口径改为 "degrades or fails to improve in 29 of 30"（比裸 "fails" 稳）；给最反直觉两证据（a=g 1.57×、scratch +0.558 vs +0.035）。
- **第 5 段（机制+可证伪）**：2/192 维、190 维冗余编码、旁路；可证伪预言（加权应消危害）→ 实测只回持平 → "decodable but not load-bearing"。
- **第 6 段（结论二+建设性）**：protocol lever 2.2–8.3×（同码同指标同种子，正对照排除 pipeline/指标伪影）；建设性收尾（补"如何演化"而非"状态是什么"）；extrinsic 必要不充分；末句跨 backbone 复现。
- **contributions**：四条保留，与新正文一致（bullet 1 已 "systematic"）。

页数仍 12 页（更紧凑），编译零错误、无 undefined 引用。Figure 1（fig:teaser，presence-not-use）保留在首页。

## 11. Figure 1 换成架构图（2026-07-19）

用户提供 LeWM physics-injection 架构 SVG，替换原 Figure 1（fig:teaser，presence-not-use 折线图）。

- 新增 `figures/fig1_architecture.svg`（修正粘贴时的字符乱码：ẑ、≈；字体改 DejaVu Sans，系统无 Anthropic Sans）→ cairosvg 转 `fig1_architecture.pdf`（pdflatex 不支持 SVG）。装了 cairosvg/pymupdf 到 `p3_llm_env`（绝对路径 pip）。
- intro 图块换成架构图，label `fig:teaser`→`fig:arch`，新 caption（encoder→黑盒 predictor + kinematic head→拼接 ẑ→MSE/SIGReg；右栏三注入机制 Slot/Probe/Kinematic；点出 2-vs-190 旁路 crux）。
- 引用调整：para 1 去掉旧图引用；para 2 "inject physical structure...(Figure~\ref{fig:arch})" —— 图右栏 Slot/Probe/Kinematic 正好对应该句三种注入方式。
- Figure 1 落在第 1 页右上角（column 2 顶），符合 reviewer 对 Fig 1 位置的要求；单栏宽度下标签清晰。
- 旧 `fig1_thesis_presence_not_use.pdf` 文件保留未删；presence-not-use 折线图如需保留可移入 §4。

编译零错误、无 undefined 引用，仍 12 页。

## 12. 架构图下标修复 + presence 图移入 §4.3（2026-07-19）

- **下标重叠修复**：cairosvg 对 `dominant-baseline="central"` + tspan `dy` 组合有 bug，导致 z_{t+1}/o_{t+1}/ẑ_{t+1} 的下标压在基字上。改为**手动双 text 定位**（基字 anchor=middle + 下标独立 text anchor=start，各自 dominant-baseline=central、y 下移 4px）。6 个带下标标签（o_t, o_{t+1}, z_t, z_{t+1}, a_t, ẑ_{t+1}）全部修正，重转 fig1_architecture.pdf。
- **presence 图归位**：原 Figure 1（fig1_thesis_presence_not_use.pdf）放入 §4.3「Mechanism: Decodable but Not Load-Bearing」节首作总起图，label `fig:presence`（= Figure 3）。节首新增一句 "The puzzle of \S\ref{sec:intro} localizes here … decodable, but not load-bearing (Figure~\ref{fig:presence})"。caption 精简保留（present 平线 + TF 崩塌 + FR 稳住）。该节此前删掉 bypass 图后正好缺图，主题契合。
- 图编号：fig:arch=1 / fig:scan=2 / fig:presence=3 / fig:lbr=4 / fig:fr=5 / fig:ladder=6 / fig:dinowmscan=7。编译零错误、无 undefined、仍 12 页。

## 13. 新增 §3.1 Setting and Notation + 符号统一 + 腾页（2026-07-19）

按外部意见解决三个问题：§3 前向引用（192 维/H 等定义在 §4.1）、全文零公式、λ 符号重载。

- **新增 §3.1 "Setting and Notation"**（label `sec:prelim`）：LeWM 模型符号（$E_\theta$, $F_\phi$, $z_t\in\mathbb{R}^{192}$）+ 两个公式——**Eq. 1** 总损失（加权 pred loss + β·SIGReg + λ·L_phys，正式定义 $w$=slot 预测权重、$\lambda$=辅助损失权重，"two distinct dials"）；**Eq. 2** kinematic head 二阶更新（$a\in\{\mathrm{MLP},g\}$）。TF/FR 与 $H{=}8$ 的定义、PushT init 声明也移入此节。
- **§4.1 Model 段瘦身**：模型描述上移后只留析因规模（~200 runs）与种子（3072/1234/42）。
- **符号统一**：Table 1 reweight 行 "(weight 30)"→"($w{=}30$, Eq. 1)"；§3.3 Test 3 "Sweep the slot weight $w$ of Eq. 1"；§4.3 Test 3 "$w{=}1\to300$"；Figure 4 caption 注明 "axis label pos\_weight λ denotes $w$"（fig8 源脚本不在仓库、未重生成）。λ 专属辅助损失（probe λ=10/λ=1 用法不变）。
- **LBR 缩写修复**：重新生成 `fig16_scan_paper.pdf`（standalone 脚本，aff env matplotlib；数据不变），行标 "[slot] +pw30 (LBR)"→"[slot] +reweight (w=30)"、"[probe] +structpos"→"[probe] +slot"；master 脚本 storyline_figures.py 同步改。
- **腾页**：删 §4.6 Synthesis（收束句 "decodable but not load-bearing" 移到 §4.5 末尾）；Related Work Phys-JEPA 段 5 句压 1 句（38%、2/192 论证保留在 §4.2/§4.3）；§3.3 首句去掉与 §3.1 重复的 TF/FR 括号定义。
- 编译零错误、无 undefined、无 Overfull，仍 12 页。Eq/sec 标签：eq:loss=1、eq:kin=2、sec:prelim=3.1。

## 14. 公式压宽 + 动作/加速度符号解冲突（2026-07-19）

**问题一：公式过宽导致 tag 换行。** AAAI 单栏 3.3in 放不下公式本体 + 右侧编号时，LaTeX 会把 (1)/(2) 挪到下一行（非错误，但每个公式白占一行）。

- **Eq (1)**：删掉 `\underbrace{}`（宽高双重元凶），主式压成 $\mathcal{L}=\mathcal{L}_{\mathrm{pred}}+\beta\,\mathrm{SIGReg}(z)+\lambda\,\mathcal{L}_{\mathrm{phys}}$，$\mathcal{L}_{\mathrm{pred}}$ 的加权求和形式（含 $w_i$ 定义）改为紧随其后的行内公式。
- **Eq (2)**：把 $a\in\{\cdot\}$ 从主行移到正文（"the acceleration $a$ is either a learned MLP or a single learnable constant---the strict $a{=}g$ head"）。
- 结果：两式编号均回到同行右对齐，各省一行。

**问题二：$a$ 重载（加速度 vs 动作 $a_t$）。** 统计后取顾问的 Option B——`a=g` 全文出现 7 处（intro / Table 1 / Table 2 caption / §4.2×2 / 附录×2），是反复使用的物理招牌；动作数学符号仅 3 处。故**保留加速度 $a$ 与 `a=g`，把动作改为 $u_t$**（控制论惯例）：
- §3.1：$F_\phi(z_{\le t},u_t)$、$\mathrm{MLP}(z_t,u_t)$；
- Figure 1 caption：$(z_t,a_t)$ → "the latent history and action $u_t$"；
- Figure 1 SVG：动作箭头标签 `a_t` → `u_t`（右栏 Kinematic 行 "ẑ = z + v + a" 中的 $a$ 是加速度，保持不变），重转 PDF。
- Table 1 "learned $a$"、全文 "strict $a{=}g$" 一律不动。

编译零错误、无 undefined、无 Overfull，仍 12 页。

## 15. 色盲无障碍改色（2026-07-19）

对全部 7 张图做 deuteranopia / protanopia 模拟（Machado 2009 矩阵，脚本 `scratchpad/cvd_sim.py`），发现 3 张不合格：

- **Figure 2 / Figure 7（两张注入扫描热力图）**：Fig 2 用 green–white–vermillion 发散色，Fig 7 用 `RdYlGn_r`。红绿色盲下绿色→灰，**唯一的真实增益格（0.76×）褪成灰色**，与"持平"格无法区分。改为 **blue–white–vermillion**（Okabe-Ito `#0072B2` / `#D55E00`，CVD-safe 发散标准配色）。两图统一。
- **Figure 1（架构图）**：绿色注入模块→灰白，与 190-d 黑盒/帧框同色，而图注恰恰指示读者"看绿色"。改为：**注入=琥珀** (`rgb(252,232,205)`/`rgb(176,88,10)`)、**损失项(SIGReg/MSE)降为中性灰**（避免与琥珀撞色）、模型蓝紫与数据米色不变。
- 其余 4 张（presence、dose-response、free-rollout、horizon ladder）通过：主对比均为蓝/橙，绿线仅作参考线或有独立标记形状。

**正文颜色描述同步**：Fig 2 caption "Red = worse, white = parity, green = better" → "Vermillion = worse, white = parity, blue = better"；Fig 1 caption "\emph{Green} marks..." → "\emph{Amber} marks..."；Fig 7 图内标题 "red = worse" → "vermillion = worse"。附录无其他颜色描述。

### ⚠️ 事故与恢复：fig17 一度被覆盖为空图

重跑 `fig17_dinowm_scan.py` 时脚本读不到 `/data1/` 上的 run 日志，静默生成了 30 格全 `--` 的空表并**覆盖了两份副本**（源目录 + paper/figures），zip 内无备份。

- **恢复**：改色前做 CVD 模拟时已渲染过原图，从该 PNG 中读回全部 30 格数值与 † 标记，写 `scratchpad/restore_fig17.py` 硬编码重建。**校验：重建图算出 10 worse / 17 parity / 3 better，与附录 caption 声称的 10/17/3 完全一致。**
- **防复发**：给 `fig17_dinowm_scan.py` 加了 guard——`if np.all(np.isnan(M)): raise SystemExit(...)`，无数据时拒绝写入而非覆盖（已验证生效）。

编译零错误、无 undefined、无 Overfull，仍 12 页；PDF 内 fig17 已确认为真实数值（0 个 `--`）。

---

## 验证

- 每步 `pdflatex` × 2，`grep "^!" main.log` 均为空（零错误）。
- `grep -rn photorealistic sections/` 为空；`grep -c labelled sections/*.tex` 全 0。
- 最终 `main.pdf` 382,710 bytes，2026-07-17 20:41。

## 遗留 / 待讨论

- contributions 第一条仍用旧轴命名口径（未出现 state/evolution 字样，暂不冲突）；如需改为 "physical quantities / physical laws" 口径可再改。
- 摘要不再提 frozen-DINOv2 复现与 extrinsic "necessary but not sufficient"——若想保留跨模型卖点，可在摘要倒数第二句后补半句。
- 摘要结尾 "should" 句的防御依赖第 3 项的 conclusion 让步；若审稿人仍攻此句，备选措辞 "would have to"（虚拟语气）在讨论中已给出。
