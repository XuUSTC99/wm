# 论据:物理注入全扫描 —— 5 家族 × 10 臂 × 3 域 = 30 格,29 格不优于 baseline

> # 🎯 一句话结论
> **把"往共享 latent 注入物理"的设计空间扫满——5 个机制家族、含变体共 10 臂、每臂 3 域 = 30 格,沿"硬/软 × 约束状态/演化 × 有/无标签"三正交轴铺开。判决指标(nMSE↓)下:25 格明确差于纯 free-rollout baseline、4 格持平(1 格恰好持平 + 3 格三种子后回落 = 种子噪声)、仅 1 格真小赢(posvel·parabola −0.026,单域单分区、量级远小于 free-rollout 的 ×2–3)。→ 物理结构不是通用杠杆,且"换编码方式会不会好"在主轴上已被实测否定。**

**对应主张**:[01_results_ledger.md](../01_results_ledger.md) **C4** / [06_storyline.md](../06_storyline.md) 步4-5(发现一)
**判读**:每域取最可靠判决分区——uniform/collision **both-OOD**、parabola **r/m-OOD**(both-OOD 有 h28 除零爆点,见 [evaluation_traps 陷阱4](evaluation_traps.md));nMSE↓;⚠️ 除注明外**单种子(3072)**。baseline(纯 free-rollout)= uniform 0.131 / parabola(r/m) 0.127 / collision 0.393。

---

## 1. 热力图(一眼看全 30 格)

![](../figures/fig16_physics_injection_scan.png)

**读图**:纵向 10 行 = 10 个注入臂(按 5 家族分组);横向 3 列 = 三域判决分区。格内数字 = 该臂在该域的 **nMSE / baseline 倍数**及原始 nMSE(括号内);**颜色 = 该倍数本身**——红=更差、白=持平、绿=更好。**标记**:`†` = 该格已跑三种子,数字/颜色用**三种子均值**(其余 26 格为单种子 3072);`✓` = 唯一真提升。

**几眼结论**:**几乎全红**——整列 collision(1.33–1.66×)最狠,冲量域拒绝一切物理结构;**唯一绿格 = posvel·parabola 0.76×(†✓,三种子确证)**;另三个 †(+pw30·uniform、probe·parabola、probe+structpos·uniform)单种子 <1、**三种子后全部落回白色持平**(见下方 ⁿ 脚注,两格均值甚至略高于 baseline)。→ 图上颜色是**三种子实测的最佳估计**,不是按结论强行上色。

**为何 26 格只跑单种子仍可下判决**:这些格子普遍高出 baseline **5–20 倍种子标准差**(σ≈0.007–0.014),远超噪声带;真正贴近 baseline、可能被种子翻盘的 4 格已全部补到三种子(即 4 个 †)。

## 2. 5 家族 → 10 臂:每臂是什么

| 家族 | 臂 | 一句话原理 | 轴定位(硬软·状态演化·标签) |
|---|---|---|---|
| **① 固定 slot** | structpos | 硬钉前 2 维 = 真实位置(`emb[:,:2]≈proprio`,无读出头) | 硬·状态·有标签 |
| | +pw30(LBR) | 上面 + pos_weight=30 承重加权(让 slot 在 pred_loss 里占比↑) | 硬·状态·有标签 |
| | +velocity(posvel) | 位置**和速度**都进 slot(×pw30) | 硬·状态·有标签 |
| **② 深监督 probe** | probe | 加线性头从 latent 读出位置(不固定维度,`probe_head(emb)≈proprio`) | 软·状态·有标签 |
| | +structpos | 软 probe + 硬 structpos 组合 | 软+硬·状态·有标签 |
| **③ 运动学头 dynamics** | free MLP | 位置 slot 挂显式方程 `z+v+a`,a=自由小网络(会过拟合) | 硬·演化·有标签 |
| | strict a=g | 同上但 a=可学重力常数(严格 PIWM 形式) | 硬·演化·有标签 |
| **④ consistency** | consistency | 不假设 a 形式,只要求预测 rollout 位置的差分速度=真值速度(为 collision 冲量设计) | 软·演化·有标签 |
| **⑤ label-free** | label-free | 不钉真值,只要求 slot 按二阶动力学**平滑演化**、指望位置自组织进 slot(动机=纯视频无 proprio) | 软·演化·**无标签** |
| | grounded | label-free 同结构 + 额外钉真值(有标签对照) | 硬·演化·有标签 |

**三正交轴闭合**:硬(structpos/dyn)↔软(probe/cons);状态(structpos/probe)↔演化(dyn/cons/label-free);有标签(9 臂)↔无标签(label-free)。→ 覆盖"往共享 latent 注入物理"的整个设计空间。

## 3. 全 30 格数据(判决分区 nMSE↓)

| 臂 | uniform(both) | parabola(r/m) | collision(both) |
|---|---|---|---|
| **baseline(纯 FR)** | **0.131** | **0.127** | **0.393** |
| structpos | 0.183 ❌ | 0.156 ❌ | 0.651 ❌ |
| +pw30(LBR) | 0.114 ≈ⁿ | 0.160 ❌ | 0.596 ❌ |
| +velocity(posvel) | 0.207 ❌ | **0.093 ✅** | 0.621 ❌ |
| probe | 0.167 ❌ | 0.115 ≈ⁿ | 0.647 ❌ |
| probe+structpos | 0.125 ≈ⁿ | 0.127 = | 0.607 ❌ |
| dyn free MLP | 0.155 ❌ | 0.178 ❌ | 0.560 ❌ |
| dyn strict a=g | 0.206 ❌ | 0.173 ❌ | 0.559 ❌ |
| consistency | 0.151 ❌ | 0.147 ❌ | 0.640 ❌ |
| label-free | 0.171 ❌ | 0.172 ❌ | 0.653 ❌ |
| grounded | 0.166 ❌ | 0.156 ❌ | 0.524 ❌ |

**❌ 差 / = 恰好持平 / ≈ 单种子<1 但三种子=持平 / ✅ 真提升**。计数:**25 ❌ + 1 = + 3 ≈ + 1 ✅ = 30 格** → **29 格不优于 baseline、1 格真小赢**。
(⚠️ 旧稿写的"26 差 / 3 平 / 1 赢"有误:把 probe+structpos·parabola 的 0.127 vs baseline 0.127 = **恰好 1.00× 持平**误计成"变差";结论方向不变,数字已订正。)

**ⁿ 三个 ≈ 格的种子真相(2026-07-15 三种子全部实测坐实,单种子<baseline 均为抽到好种子、非真提升)**:
- **+pw30·uniform**:单种子 0.114 → 三种子 **0.132±0.014**(0.114/0.135/0.147)vs baseline 0.136±0.007 → 持平。
- **probe·parabola**:单种子 0.115 → 三种子 **0.137±0.027**(0.115/0.176/0.121)vs baseline(r/m)0.122±0.005 → 持平(**三种子均值反而略高于 baseline**)。
- **probe+structpos·uniform**:单种子 0.125 → 三种子 **0.141±0.014**(0.125/0.141/0.159)vs baseline 0.136±0.007 → 持平(**均值亦略高**)。
- 结论:**三格三种子后全部落回持平(两格均值甚至略差)**,单种子的"净超"是种子噪声。热力图据此把这三格上白色(持平)——是三种子实证、非按结论强行上色。源:`structdyn_eval/rollout_{parabola_probeF2_fr,uniform_probeF2_structpos_pw30_fr}_s{1234,42}.log`(2026-07-15 补跑)。
- **probe·parabola 0.115**:落在 baseline 三种子区间(0.115–0.127)**下沿 = 噪声**。
- **probe+structpos·uniform 0.125**:同 baseline 噪声带。

## 4. 分析

### 4.1 唯一真提升 = posvel·parabola,是机制签名不是可用方法
posvel 在 parabola 的 r/m-OOD:单种子 0.093、**三种子 0.096(0.093/0.091/0.104)vs baseline 0.122±0.007,逐种子 3:0、区间零重叠,−0.026 真实**。但:①**单域单分区**——同一编码(速度进 slot)在 uniform +0.076、collision +0.228 反而变差;②量级 −0.026 **远小于** free-rollout 的 ×2–3;③**需预知动力学**(速度在抛体里是线性驱动量、可外推;匀速是冗余常数、碰撞是跳变)。→ **是"① slot 占比高到不被旁路绕过 + ② 编进去的量在该域可外推"两条同时满足才有用的机制签名,不是通用先验**(详见 [why_physics_structure_fails 层1](why_physics_structure_fails.md))。

### 4.2 三条跨臂规律
1. **collision 整列全红(1.33–1.66×)、最狠**:冲量域连 grounded(有完美标签,least-bad 1.33×)都伤——富动力学/不连续域拒绝一切平滑物理结构。
2. **"演化"类不比"状态"类好**:dynamics/consistency/label-free(演化)与 structpos/probe(状态)同样全败——问题不在"钉状态还是钉演化"。
3. **标签有无、软硬都不救**:label-free(无标签)vs grounded(有标签、同结构)同域同向差;硬(structpos)vs 软(probe)同向差 → 失效与"标签/软硬"无关,是**架构性**(物理占比低 + 黑盒旁路,见 [why_physics_structure_fails 层2](why_physics_structure_fails.md))。

### 4.3 对"换个 intrinsic 编码会不会好"的回答
30 格已沿三正交轴铺满、含**正确的物理形式(a=g)**、含**承重加权**、含**无标签自组织**——全败(仅 1 格匹配动力学的小赢)。→ "换编码方式"在主轴上被实测否定;要突破须**改架构堵旁路(extrinsic)**,不是换 intrinsic 编码(详见 [why_physics_structure_fails 层3](why_physics_structure_fails.md))。

## 5. 数据来源(逐臂,可复现)

| 臂 | log(`/data1/likun-share/junjxu/runs/…`) |
|---|---|
| baseline | `structdyn_eval/rollout_{uniform_motion,parabola,collision}_baseline_fr_id1k.log` |
| structpos | `aaai_p0/rollout_{parabola,collision}_structpos_fr_id1k.log`、`structdyn_eval/rollout_uniform_motion_structpos_fr_id1k.log` |
| +pw30 | `structdyn_eval/rollout_{域}_structpos_fr_pw30_id1k.log`(3 种子 `aaai_p0/…_s{1234,42}`) |
| +velocity(posvel) | `{aaai_p0,structdyn_eval}/rollout_{域}_structposvel_pw30*.log`(par 3 种子) |
| probe | `structdyn_eval/rollout_{域}_probeF2_fr.log` |
| probe+structpos | `structdyn_eval/rollout_{域}_probeF2_structpos_pw30_fr.log` |
| dyn free MLP | `structdyn_eval/rollout_{uniform,parabola}_dyn_mlp_fr_id1k.log`、`…_collision_structdyn_fr_id1k.log` |
| dyn strict a=g | `structdyn_eval/rollout_{域}_piwm_const_fr_id1k.log` |
| consistency | `consistency_eval/rollout_uniform_cons_B_v1.log`、`structdyn_eval/rollout_parabola_structpos_cons1p0acc_id1k.log`、`…_collision_structpos_cons1p0_id1k.log`(⚠️域间用了不同 cons 变体,collision cons 全族 0.549–0.653 均>baseline,结论不变) |
| label-free | `structdyn_eval/rollout_{域}_labelfree_const_id1k.log` |
| grounded | `{structdyn_eval,aaai_p0}/rollout_{域}_grounded_const_id1k.log` |

图脚本:[storyline_figures.py](../figures/storyline_figures.py) fig16(数字内嵌)。⚠️ 全表单种子(3072),已补三种子的:+pw30、posvel·par(见上)。
