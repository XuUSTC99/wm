# 【来自 lewm 会话】PIWM 官方 baseline 首次在 phyworld 出数（回应"无外部 baseline"P2-4）

**2026-07-13**。完整表/数据源/图见 [../piwm_baseline/PLAN.md](../piwm_baseline/PLAN.md) §5-6，图 [../figures/fig2_piwm_vs_lewm.png](../figures/fig2_piwm_vs_lewm.png)。

## 一句话
官方 PIWM（extrinsic-conti，忠实移植，最有利 δ=0）在 phyworld 学到**正确物理**（uniform s=1.0/g=0；parabola g_y=−0.028≈真值−0.025），**ID/v-OOD 位置 ρ 比 LeWM 还高**，但 **r/m-OOD、both-OOD 崩**（uniform r/m ρ 0.33、both 0.48；LeWM 0.89/0.87）。

## 对论文的用法
- **补上"无外部方法 baseline"短板**（P2-4 从 future work 变已做）。
- **强化主线**：物理结构买到干净 ID 物理外推、**买不到 OOD 鲁棒**——崩的是 VAE 编码器（没见过的球尺寸编错位置），不是方程。是 decodable≠load-bearing 的**编码器版**证据。
- 建议进 §4（外部 baseline 对照）或 Related/Experiments，配 fig2 grouped-bar。
- **公平性口径**（写进正文脚注）：LeWM 侧 ρ 是 probe（eval 集 80/20 训）读 rolled-out latent，PIWM 是直接输出，协议偏袒 LeWM；即便如此 PIWM ID/v 仍更高，OOD 崩非协议造成。nMSE 两者归一化基不同不可比，只用 ρ。

## 数据源
train/eval/ckpt: `/data1/likun-share/junjxu/runs/piwm_baseline/{redyn_*.log,eval_*_d0.json,ckpts/}`；代码 `PIWM/phyworld_port/`。
