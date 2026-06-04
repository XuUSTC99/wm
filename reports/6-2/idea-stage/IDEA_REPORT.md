# Idea Discovery Report — 让 LeWM 更"遵守物理规律"

**方向**：基于 [piwm_three_domains.md](../piwm_three_domains.md) 的结果 + "方向1：物理结构化 latent predictor(PIWM 原则2 搬到 JEPA latent)"，找更好的创新点
**日期**：2026-06-01
**Pipeline 执行说明**：本会话**无 Codex/gpt-5.5 MCP** → 跳过所有跨模型评审(idea-creator/research-review 的 GPT 复核);**未跑 GPU pilot**(predictor 侧方法尚未实现,且不擅自占 8 GPU-hr)。本报告 = 单模型 idea 生成 + web-search 文献 grounding + 轻量 novelty 框定。正式投入前请用 /novelty-check 对选定 idea 复核。

---

## 执行摘要

把"方向1"原样做(HNN/已知方程套到 LeWM latent)**novelty 低**——被 PIWM(已知方程+弱监督)和 HNN/LNN(state 空间)夹住。**真正的开口**:利用我们独有的两个结果——(a) rollout 多步漂移、OOD 更快崩;(b) within-traj 方差能区分"哪个物理量守恒"——做一个**不需要已知方程、非侵入式(冻结 encoder)的"守恒一致 rollout"**。推荐 idea: **🏆 Idea 1 (PhysConsist-Rollout)**。

---

## 文献地形(Phase 1，web-search grounding)

| 簇 | 代表 | 跟我们的关系 |
|---|---|---|
| 结构化动力学(state 空间) | HNN / LNN / Hamiltonian Generative Net / Koopman+conservation | 成熟,但都在**已知物理变量/能量形式 + state 空间**;不是自监督 latent |
| 物理可解释世界模型 | **PIWM** [2412.12870] | 已做"已知方程 + 学参数"的 latent dynamics,但**要弱监督物理变量 + 知道方程**(bicycle/cartpole) |
| 自监督 JEPA WM | **LeWM** [2603.19312] | 我们的 backbone;predictor 是**黑箱**,无物理结构 |
| 评估侧(热) | **Observer Effect** [2602.12218]、Interpreting Physics [2602.07050]、Probing Latent World [2603.20327] | adaptation 损伤 latent physics;probe 物理量。**只诊断,不给修法** |
| 结构化 latent transition | Homomorphic latent dynamics [2603.20048]、Latent Particle WM [2603.04553] | action 作平滑变换 / 物体中心,**不聚焦守恒律** |

**结构性 gap**:没人做"**从冻结自监督 latent 里发现守恒量(不靠已知方程)→ rollout 时非侵入地强制守恒 → 修长程 OOD 漂移**"。PIWM 要已知方程,Observer Effect 只诊断不修。

---

## Ranked Ideas

### 🏆 Idea 1 — PhysConsist-Rollout：自监督守恒量发现 + 非侵入守恒投影 rollout
**一句话**:冻结 LeWM encoder(它已 ρ>0.9 OOD 编码物理),**用 within-traj 方差从 latent 里自动发现"该守恒的量"**,在 ARPredictor rollout 的每一步把预测 latent **投影回守恒流形**(constraint projection),修我们测出的长程 OOD 漂移。
- **为什么对症**:直接攻 rollout_results 的核心负结果(1步准、多步漂、OOD 崩),且 §5 的 within-traj std 正好当"发现哪个量守恒"的信号——**把我们已有结果串成 motivation→method 闭环**。
- **Novelty hook**:vs PIWM=不需要已知方程/弱监督;vs HNN=latent 空间 + 守恒量是**发现的不是给定的**;vs Observer Effect=给**修法**而非诊断(且"非侵入/冻结 encoder"正好呼应它"adaptation 伤 latent"的告诫)。
- **Novelty(web-search)**:MEDIUM-HIGH。无直接对应;最近邻 PIWM(已知方程)+ "learn arbitrary conservation laws" 类(state 空间)。**需 /novelty-check 复核**。
- **最小 pilot(<2 GPU-hr)**:parabola/collision 上,(a) 从冻结 latent 估每条 traj 的候选不变量(probe 解出的量里挑 within-traj std≈0 的),(b) rollout 每步加一个"把 probe 解出的不变量拉回初值"的投影/正则,(c) 比长程 cos / OOD rollout MSE vs 原版 LeWM。预期信号:h≥8 的 cos 不再崩。
- **风险**:"discover conservation law" 本身有人做过(Koopman 类),delta 全靠"自监督 latent + 非侵入 + OOD physics"的组合;投影可能与 JEPA latent 几何冲突。

### Idea 2 — Non-Invasive Structured Predictor：冻结 encoder，只把 predictor 换成结构化 latent ODE
**一句话**:encoder 冻结,**只**把黑箱 ARPredictor 换成"在 probe 物理子空间上的结构化 latent ODE(已知/半已知形式 + 学参数)",验证能否在不损伤表征(Observer Effect)的前提下修长程漂移。
- **Novelty**:MEDIUM。比 Idea 1 更接近 PIWM 原则2,但"非侵入(冻结 encoder)+ 只结构化 predictor"是没被强调的消融轴。可作 Idea 1 的对照臂/baseline。
- **风险**:跟 PIWM 重叠度较高;需要"半已知方程"——又回到 PIWM 的前提。

### Idea 3 — Drift = Physics-Violation 的自校正 rollout(predictor-corrector)
**一句话**:把 AR 漂移显式建模成"守恒残差累积",每步用守恒残差做**校正项**(predictor 给方向,projection 给约束),类数值积分的 predictor-corrector。
- **Novelty**:MEDIUM。机制新颖;但跟 Idea 1 的投影高度重合,更像 Idea 1 的一种实现变体。
- 建议:并入 Idea 1 作为"硬投影 vs 软残差校正"的消融。

### Idea 4 — within-traj 方差自适应的多尺度结构(novelty-check 挖的种子)
**一句话**:按每个物理量的 within-traj 方差自动选约束类型(std≈0→守恒约束;线性变→学一阶 ODE;跳变→事件检测)。
- **Novelty**:MEDIUM,但**偏窄、偏 incremental**,reviewer 易说 "expected"。
- 建议:作为 Idea 1 的"如何选该守恒哪个量"的子模块,不单独主推。

### Idea 5(对照,非主推)— surprise/violation-of-expectation 升级成"哪条物理律被违反"
LeWM 已有 surprise 检测;扩成 law-aware(指出违反动量/重力/惯性中的哪条)。偏应用,novelty 一般。

---

## 推荐路径

**主推 Idea 1**,把 Idea 2(非侵入 structured predictor)当 baseline 对照、Idea 3(软校正)/Idea 4(自适应选量)当消融。一篇完整论文弧线:

> **"Physics-Consistent Rollout for Self-Supervised Latent World Models without Known Equations"**
> - Motivation/分析(✅ 已有):rollout 1步准多步漂、OOD 更崩(rollout_results)+ within-traj 方差区分守恒量(piwm_three_domains §5)
> - Method(待做):latent 不变量发现 + 非侵入守恒投影 rollout
> - 评估:长程 OOD rollout MSE / cos,跨 3+ PhyWorld 域 + 真实视频
> - 对照:vanilla LeWM、PIWM(已知方程上界)、deep-sup(我们做过,改 readout 不改演化→无效,正好反衬)

---

## 下一步(可执行,不依赖 Codex)

1. **/novelty-check** 复核 Idea 1 的精确 claim(尤其 vs "learn conservation laws" 系 + PIWM)——**最关键,先做**
2. **/arxiv** 精读:PIWM 全文(原则2 实现细节)、HNN/LNN、Koopman-conservation、Observer Effect(借它的 OOD physics 评估协议)
3. 写 RESEARCH_BRIEF.md(problem anchor:长程 OOD 漂移 = 物理律违反累积),再来跑可执行版 idea-refine
4. Idea 1 最小 pilot(<2 GPU-hr):在现有 parabola/collision ckpt + rollout 脚本上加守恒投影,测 h≥8 cos

---

## 跳过/未做(诚实标注)

- ❌ 跨模型评审(idea-creator GPT-5.4 / research-review gpt-5.5):本会话无 Codex MCP
- ❌ GPU pilots:方法未实现 + 未获 8 GPU-hr 授权;上面给了 pilot 设计待你批
- ❌ /research-lit 深度多源 / /render-html:子 skill 在本会话不可用,用 web-search 替代了 Phase 1
- ⚠️ 所有 arXiv ID 来自本次搜索返回,非凭记忆;[2602.07050]/[2603.20327]/[2603.20048]/[2603.04553] 仅见搜索标题,正文待核

**Sources**: [2412.12870 PIWM](https://arxiv.org/abs/2412.12870) · [2603.19312 LeWM](https://arxiv.org/html/2603.19312v1) · [2602.12218 Observer Effect](https://arxiv.org/abs/2602.12218) · [Hamiltonian Generative Networks 1909.13789](https://arxiv.org/pdf/1909.13789) · [LNN (Cranmer)](https://astroautomata.com/data/lnn.pdf) · [2603.20048 Homomorphic latent dynamics](https://arxiv.org/pdf/2603.20048) · [2603.04553 Latent Particle WM](https://arxiv.org/abs/2603.04553)
