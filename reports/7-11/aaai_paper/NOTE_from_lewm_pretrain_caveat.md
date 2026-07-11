# 【来自 lewm 会话】pretrain 2×2 数据核对 + 一个必须纠正的 caveat

**2026-07-11，lewm 会话（a29d4510）留给 aaai_paper 会话**
关于你 [01_results_ledger.md](01_results_ledger.md) L77-91 的 pretrain 2×2 / P0-3 数字。

## 好消息：结论稳
C4 "物理 from-scratch 也伤" **三域干净 scratch 全部 Δ>0，成立**。PIWM"要从头训"辩护封死。放心用。

## ⚠️ 必须纠正：parabola both-OOD 是数值爆点，别当 headline

你 L77-83 引的 **parabola scratch_on both-OOD=1.201 / Δ+0.550 / "8.7×" / "与 uniform +0.558 惊人一致"**——**parabola 这个 1.201 是被 h28 数值爆点污染的**：
- parabola scratch_on **h28 nMSE = 1,976,290**（197万，除零：球出框/目标方差→0），但 **h28 cos 仍 0.91**。both-OOD 聚合被这几条轨迹拉爆。
- 所以 "+0.550" 不是干净信号；"与 uniform +0.558 惊人一致" 是**巧合**（uniform 是真、parabola 是 artifact）。
- **审稿人一算 nMSE 就会发现 h28 发散，这个具体数字会被打穿。**（也正好又是你 §6"cos-nMSE 陷阱"的实例——但这里是 nMSE 自己爆，方向相反，别混用。）

## 用这些干净数字替换（我已逐一核 h28 无爆点）

| 域 | scratch off | scratch on | Δ_scratch | h28 nMSE | 干净? |
|---|---|---|---|---|---|
| **uniform** | 0.192 | 0.750 | **+0.558** | 1.27 | ✅ 用这个当 headline |
| collision | 0.538 | 0.635 | +0.097 | 1.65 | ✅ 干净 |
| parabola | 0.343(r/m) | 0.375(r/m) | +0.03(r/m) | — | both-OOD 弃用，走 r/m-OOD |

对照 Δ_pusht（物理后训练效应）：uniform +0.035、parabola(r/m) +0.07。

## 建议改法（3 选 1）
1. **最稳**：headline 用 **uniform**（scratch off 0.192→on 0.750，Δ+0.558，干净、且"基线越干净物理伤越狠"的最佳案例）；parabola 走 r/m-OOD。
2. Table 2 三域统一用 **r/m-OOD nMSE**（唯一各域都无爆点），一致可比。
3. 若保留 both-OOD：parabola 那格标注"h28 数值发散，见 r/m-OOD"。

**"8.7×" 这个倍数别写**（建立在 artifact 上）；改成"物理在 from-scratch 下伤得更狠，uniform 上 Δ 从后训练 +0.035 放大到 +0.558"——真实、且够狠。

## 出处
`/data1/likun-share/junjxu/runs/pretrain_physics/rollout_pp2_par_*.log`（parabola 120ep）
`/data1/likun-share/junjxu/runs/aaai_p0/rollout_pp2_{um,col}_scratch_*.log`（uniform/collision）
分析全文：[../pretrain_physics/EXPERIMENT_PLAN.md](../pretrain_physics/EXPERIMENT_PLAN.md) §6-7
