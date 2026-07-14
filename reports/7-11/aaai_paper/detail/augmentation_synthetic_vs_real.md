# 论据:数据增广是合成域最强杠杆,但合成→真实反转 100×

> # 🎯 一句话结论
> **同一个 appearance 增广,在简单合成域是最强 OOD 杠杆(−48~63%),搬到照片级仿真 nMSE 崩 100×——真实场景外观携带物理信息(摩擦/质量/材质)、不是可抹掉的 nuisance,所以 appearance 增广是反面边界警示、不是正面方法(几何 scale 增广不反转,可留)。**

**对应主张**:[01_results_ledger.md](../01_results_ledger.md) **C3** / [06_storyline.md](../06_storyline.md) 发现二（增广边界警示）
**配图**:[../figures/fig4_aug_synthetic_vs_real.png](../figures/fig4_aug_synthetic_vs_real.png)

![](../figures/fig4_aug_synthetic_vs_real.png)

---

## 1. 合成域(PhyWorld):增广最强,超过所有物理结构方法

both-OOD nMSE↓（parabola 有 r/m 佐证）:

| 域 | 基线(FR) | 最优增广 | 降幅 |
|---|---|---|---|
| uniform | 0.131 | **0.068**（appearance 0.5） | −48% |
| parabola | 0.313（r/m 佐证 0.127→0.065） | **0.115**（appearance 0.5） | −63% |
| collision | 0.393 | **0.172**（scale 0.5 + np20，3 种子 0.172±0.004 ✅✅） | −56% |

- 超过最好的物理结构配置（uniform aug 0.068 < 物理最好 0.109）。
- **交互非平凡**:appearance×np20 冲突（0.472 ❌）、scale×np20 最佳叠加（0.208 ✅）、三重组合更差（0.253）；强度甜点 0.5。
- **时序增广证伪**:temporal stride 治不了 v-OOD（0.556 还不如 appearance 的 0.376）。
- 源:[general_augmentation.md](../../6-24/final/general_augmentation.md)、[optimization_plan.md](../../6-24/final/optimization_plan.md)。

## 2. 真实反转(Physion++ 直训):appearance 0.5 nMSE 崩 100×

base(pp_fr) → app05(叠 aug.appearance=0.5),epoch20:

| 场景 / horizon | nMSE base→app05 | 退化 |
|---|---|---|
| **friction_collision** | **0.062 → 6.44** | **~104×** |
| mass_collision | 0.083 → 0.955 | 11× |
| bouncy_wall | 0.094 → 0.828 | 8.8× |
| h64（整体） | 0.280 → 0.311 | 退化 |

- **为什么反转**:PhyWorld 的球外观是纯 nuisance（抹掉无损物理）;Physion++ 的材质/亮度**编码了摩擦/质量**,抹掉 = 抹掉物理输入。
- 源:`/data1/likun-share/junjxu/runs/physionpp/eval_pp_fr_e20.log`（base）、`eval_pp_fr_app05_e20.log`（app05）;[physionpp_ood_longhorizon.md §3.5](../../physion/physionpp_ood_longhorizon.md)。

## 3. 迁移侧也无用

phyworld→Physion zero-shot:appearance 0.5 = 0.597 < free-rollout 0.603 < random 0.607。增广突不破迁移天花板。源:[transfer_improvement_report.md §2](../../physion/transfer_improvement_report.md)。

## 4. 论文口径

- **正面框架 = 发现支配变量**（free-rollout / horizon / scale），appearance 增广**不当正面方法卖**。
- appearance 的反转本身是一条贡献:**合成→真实的边界警示**——社区在简单合成 benchmark 上得到的增广收益不能想当然迁移到真实数据。
- ⚠️ friction_collision 的 100× 是"增广伤真实"（此处 cos 也降,不是 cos 陷阱）;真正的 cos 陷阱实例（cos 升而 nMSE 崩）见 [evaluation_traps.md](evaluation_traps.md)（deform_clothhit / h64）。
