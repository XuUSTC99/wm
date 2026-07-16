# 跨模型验证：第二个 JEPA 实例（DINO-WM 风格）复现两大发现

**日期**：2026-07-16　**规模**：68 run（全部 setsid、零失败）
**动机**：论文主张"物理注入在共享 latent JEPA 上失效、训练协议才是杠杆"。只在自研 LeWM 上验证会被审稿人一句"你只测了一个模型"打掉 → 换第二个 JEPA 实例复跑核心矩阵。

## 0. 模型与实现

**dinowm = DINO-WM / V-JEPA2-AC 风格**：冻结 `facebook/dinov2-small`（384-d CLS，SSL 预训练）+ 可训练 projector(384→192 adapter) + **与 LeWM 完全相同**的 ARPredictor / losses / FR-TF 开关 / eval 脚本。
→ 与 LeWM 的差异：backbone 出身（DINOv2 SSL vs pusht JEPA）、**encoder 冻结 vs 可训练**、384 vs 192 维。这是受控的 cross-backbone 消融（只换 encoder，下游一行不改）。
实现：`le-wm/train.py` 的 `+encoder_type=dinov2 +freeze_encoder=true`；队列 `run_dinowm_queue{,2,3,4,5}.sh`；log `/data1/likun-share/junjxu/runs/dinowm/`。

**前提验证**：冻结 DINOv2 的 CLS，位置线性可解码 REAL-emb **ρ≈0.93** → "latent 已编码物理状态"这个立论前提在第二个 backbone 同样成立，注入的确实是"latent 已有的状态"。

## 1. 发现二（free-rollout 是主升力）：**跨模型完全复现** ✅✅

判决分区同论文（uniform/collision→both-OOD，parabola→r/m-OOD，physion→h64），3 种子 mean±std：

| 域 | TF | FR | 倍数 | LeWM 对照 |
|---|---|---|---|---|
| uniform | 0.594±0.064 | **0.427±0.039** | 1.39× | 2.2× |
| parabola(r/m) | 0.380±0.015 | **0.225±0.019** | 1.69× | 3.6× |
| collision | 0.803±0.034 | **0.479±0.009** | 1.68× | 2.4× |
| **Physion++(真实, h64)** | 1.030±0.028 | **0.258±0.005** | **3.99×** | 8.3× |

→ **两个架构差异显著的 JEPA 实例、3 合成域 + 1 真实数据集、3 种子，FR 全部显著赢，区间不重叠。** 倍数比 LeWM 小（冻结 encoder 的 baseline 本身更弱、天花板更低），但方向和显著性一致。**这是论文"训练协议是支配变量"主张的跨模型证据。**

## 2. 发现一（物理注入失效）：**跨模型成立，但表现形式不同** ✅

3 种子 mean±std（uniform，both-OOD；baseline FR = 0.427±0.039）：

| 臂 | dinowm 3-seed | 判决 | LeWM 对照 |
|---|---|---|---|
| [slot] structpos+pw30 | 0.347±**0.087** | **~same（error bar 重叠）** | 显著有害 |
| [probe] deep-sup f2 | 0.434±0.050 | ~same | 有害 |
| [cons] consistency | 0.428±0.047 | ~same | 有害 |
| [slot] plain（无加权） | 0.608（单种子） | **1.42× 有害** | 有害 ✅一致 |
| [dyn] slot+运动学 | 0.553 / par 0.259 / col 0.592 | **三域全害 1.15–1.29×** | 有害 ✅一致 |

collision / parabola 的核心三臂同样全部落在 baseline 的 error bar 内（col base 0.479±0.009：pw30 0.485±0.052、probe 0.493±0.007、cons 0.461±0.040；par base 0.225±0.019：pw30 0.236±0.007、probe 0.231±0.006、cons 0.227±0.006）。

**⚠️ 一个差点写错的教训**：单看 seed3072，`structpos_pw30` 在 uniform 上是 0.81×（"变好 19%"），LBR 曲线单种子版还给出 pw300→0.62×"越加权越好"——**看着像推翻论文**。补到 3 种子后真相是 **[0.300, 0.468, 0.272]、std 0.087（baseline std 的 2.2×）、error bar 完全重叠 = 噪声**。这是本项目第三次栽在"小差异+单种子"上（前两次：physion 的 cube>pusht、appearance 的 cos 假象）。

**两个模型的共同点（论文要的那句）**：**物理注入从不带来净增益**。
**差异及其机制解释**：LeWM（encoder 可训练）注入**显著有害**；dinowm（encoder 冻结）注入**无效、且方差暴涨**（std 0.087 vs baseline 0.039 = 训练不稳）。冻结让物理 loss 只能改 projector → 危害被限制在 adapter 层，但增益也无从产生——latent 已有物理（ρ=0.93），190 维旁路仍在。**这反而是机制假说的正面旁证：注入的成败取决于旁路是否存在，而非物理知识是否正确。**

## 3. C2（horizon 匹配动力学复杂度）：collision 复现，其余待种子

| 域 | np8(base) | np16 | np20 | 判决 |
|---|---|---|---|---|
| collision | 0.479 | 0.361 (0.75×) | **0.295 (0.62×)** | ✅ **吃长 rollout**（与 LeWM 一致）|
| uniform | 0.427 | 0.328 (0.77×) | — | ⚠️ 单种子，LeWM 上是"有害"，此处相反 → **不可判** |
| parabola | 0.225 | 0.245 (1.09×) | — | ⚠️ 单种子，方向与 LeWM 一致（不吃长 rollout）|

→ **collision 吃长 rollout 跨模型稳**；uniform/parabola 是单种子，且注入臂实测方差可达 0.087（足以吞掉这些差异）→ **暂不作结论**。

## 4. LBR 曲线：单种子版作废，补种子中

单种子 pw 扫描给出 0.95/0.77/0.81/0.94/0.62（pw1/10/30/100/300）——**非单调本身就是噪声特征**，且 pw30 的 3 种子 std=0.087 足以覆盖整条曲线的起伏。已启动 `queue5`（pw1/10/100/300 各补 2 种子，8 run）。**在补齐前，dinowm 的 LBR 曲线不进论文。**

## 5. 对论文的净贡献

- **可以写**："两个架构差异显著的 JEPA 实例（可训练 ViT-tiny / 冻结 DINOv2+adapter）上，free-rollout 都是显著主升力（合成 1.39–1.69×、真实 3.99×，3 种子），而 6 个物理注入机制家族都不带来净增益" → **把主张从单模型抬到 JEPA 家族级**。
- **不能写**："注入在所有 JEPA 上都有害"——dinowm 上是"无效"不是"有害"，要诚实区分。
- **加分**：有害 vs 无效的差异可归因于"旁路是否被架构限制"，与论文的 extrinsic 结论自洽。

*数据源*：`/data1/likun-share/junjxu/runs/dinowm/rollout_dinowm_*.log`；汇总脚本 `../collect_dinowm.py`（一键复现全部表格）。
