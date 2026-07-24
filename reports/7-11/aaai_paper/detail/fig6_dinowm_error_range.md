# Figure 6(DINOv2 注入扫描)的"误差范围"是怎么算的

> 问题:`Figure 6: The injection scan repeated on the frozen-DINOv2 backbone` 这张图里,
> 每格的误差 / "within noise" 到底怎么来的?

一句话:**这张热图没有传统误差棒**。每格是一个 nMSE 比值 + 一个"显著性判定",
判定用的是 **三种子 min–max 区间是否与 baseline 分离**,不是 ±std、不是 p 值、
也不是对比值卡阈值。

生成脚本:`reports/7-11/aaai_paper/figures/dino_resolve.py`(与 Figure 2 的
`scan_resolve.py` 共用同一套 verdict 规则,所以 caption 说"drawn and judged exactly
as Figure 2")。数据源:`raw_data/runs/dinowm/` 的原始 rollout log。

---

## 每一格怎么算(一个注入变体 × 一个域)

取两组 per-seed 的 nMSE:

- `d`  = 该注入格在种子 `{3072, 1234, 42}` 上的 nMSE → 3 个数(合并的 a=g 行是 4 个)
- `bd` = 同域 baseline(free rollout)在同样 3 个种子上的 nMSE → 3 个数
- 分区取"hardest clean split":`um`/`col` 用 both-OOD,`par` 用 r/m-OOD
  (对应 `PARTITION = {"um": "both-OOD", "par": "r/m-OOD", "col": "both-OOD"}`)

### ① 格子里显示的数 = 两组均值之比

```python
r = mean(d) / mean(bd)
```

例:`0.97×` = 注入的三种子均值是 baseline 三种子均值的 97%(误差低 3%)。

### ② "误差范围"其实是三种子的 min–max 全幅,判据是区间分离

```python
if   min(d) > max(bd):   verdict = "worse"    # 注入最好的种子都比 baseline 最差的种子还差
elif max(d) < min(bd):   verdict = "better"   # 注入最差的种子都比 baseline 最好的种子还好
else:                    verdict = "noise"    # 两个种子区间有重叠 → 判不开 → 画斜纹(hatched)
```

- **solid + 暖色 = worse**:注入的 `[min,max]` 完全落在 baseline `[min,max]` 之上
- **solid + 蓝色 = better**:注入的 `[min,max]` 完全落在 baseline `[min,max]` 之下
- **hatched = within noise**:两个区间**有重叠**,分不开

所以 caption 里 **"hatched = the cell's seeds overlap the baseline's"** 就是字面意思:
注入格 3 个种子的 [min,max] 和 baseline 3 个种子的 [min,max] 有交叠,就算 within-noise。

> 这个判据比 ±std 或 t 检验都**严**:要算"赢",得注入的**每一个**种子都压过
> baseline 的**每一个**种子(全分离),只要有一点重叠就退回 within-noise。

---

## 实际每格输出(跑 `python3 dino_resolve.py`)

```
[slot] structpos           um:1.17x/noise/n3  par:1.02x/noise/n3  col:0.97x/BETTER/n3
[slot] +reweight (w=30)    um:0.81x/noise/n3  par:1.05x/noise/n3  col:1.01x/noise/n3
[slot] +velocity           um:0.93x/noise/n3  par:1.01x/noise/n3  col:0.99x/noise/n3
[probe] probe              um:1.02x/noise/n3  par:1.03x/noise/n3  col:1.03x/noise/n3
[probe] +slot              um:0.91x/noise/n3  par:1.10x/noise/n3  col:1.02x/noise/n3
[dyn] free MLP             um:0.99x/noise/n3  par:1.04x/noise/n3  col:1.11x/noise/n3
[dyn] strict a=g           um:1.08x/noise/n4  par:1.15x/worse/n4  col:1.27x/worse/n4
[cons] consistency         um:1.00x/noise/n3  par:1.01x/noise/n3  col:0.96x/noise/n3
[free] label-free          um:0.96x/noise/n3  par:1.00x/noise/n3  col:1.07x/worse/n3

verdicts: 3 worse / 23 within-noise / 1 better  (26 of 27 at or below)
```

- **3 worse**:a=g 的 par(1.15×)/col(1.27×),label-free 的 col(1.07×)
- **1 better**:`[slot] structpos` 在 **collision** 上,`0.97×`
- 其余 **23 within-noise**

### 那个 0.97× 的 "below-cell" 到底怎么回事

= structpos / collision。它的 3 个种子确实**全**落在 baseline 3 个种子之下(所以判 `better`),
但 `max(注入)` 与 `min(baseline)` 只差 **0.001 nMSE**——分是分开了,只分开一根头发丝。
这就是正文/caption 里 "the one below-cell separates by a 0.001 margin at 0.97×"、
且仍算 "injection never wins" 的原因(而且视觉上发蓝的 uniform 格全是 within-noise,
再叠加 Trap 5 的 content-free 对照后连表观增益也消失)。

---

## 和 Figure 4(柱状图)的区别,别混

| | 不确定性怎么画 | "显著/分开"怎么判 |
|---|---|---|
| **Figure 4**(TF vs FR 柱状图) | 真的 **±sample std** 误差棒(3 种子样本标准差) | ±std 区间不重叠("every interval disjoint") |
| **Figure 6**(DINOv2 扫描热图) | 不画误差棒 | **min–max 种子区间**是否完全分离(斜纹 = 重叠) |

两者都基于同样那 3 个种子,只是一个用 ±1 std 区间、一个用 min–max 全幅。
后者(全分离)更保守。

---

## 一句话记忆点

> Figure 6 每格 = `mean(注入3种子) / mean(baseline3种子)`;
> 斜纹(within-noise)= 注入的三种子 min–max 与 baseline 的三种子 min–max **有重叠**;
> 实心 = 完全不重叠(全上 = worse / 全下 = better)。
> 唯一那格 "better"(structpos/col, 0.97×)只领先 0.001 nMSE,故正文仍称"从不真赢"。
