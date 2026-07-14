# 论据:真实数据(Physion++)—— zero-shot 死路、直训活路

> # 🎯 一句话结论
> **合成里学的物理搬不到真实(zero-shot 封顶=random 架构先验 0.607,无配置能超);但直接在真实仿真上训,free-rollout+长 rollout 能把长程 nMSE 打到基线的 1/19,且泛化分层——刚体动力学可迁、形变不可。**

**对应主张**:[01_results_ledger.md](../01_results_ledger.md) **C7**（+ C1/C2 真实侧）/ [06_storyline.md](../06_storyline.md) 发现二 & 结论
**配图**:[../figures/fig7_realdata_num_preds.png](../figures/fig7_realdata_num_preds.png)、[../figures/fig2_free_rollout.png](../figures/fig2_free_rollout.png)（真实侧一列）、[../figures/fig6_transfer_ceiling.png](../figures/fig6_transfer_ceiling.png)

> **术语**:Physion++ = 照片级物理仿真视频（球碰撞/多米诺/布料/斜面…,带 3D 位置+速度标注）。zero-shot = phyworld 训好直接搬去 Physion 评（测合成→真实,结论:不能）。直训 = 直接喂 Physion++ 从头训（测真实上限,结论:很好）。

![](../figures/fig7_realdata_num_preds.png)

---

## 1. zero-shot 死路:封顶 = random 先验 0.607

phyworld→Physion 所有配置 <0.607,仅 Support(+0.10)/Collide(+0.07)有真信号。见 [evaluation_traps.md](evaluation_traps.md) 陷阱2 + Fig 6。

## 2. 直训活路①:free-rollout 是长程主力（8.3×）

Physion++ 直训 h64 nMSE:**FR 0.141 vs TF 1.174（8.3×,3 种子 3v3 零重叠,cos 0.89 vs 0.50）**;物理结构（struct/cons/consacc）长程全差于纯 FR（0.168–0.266）。源:`/data1/.../runs/physionpp/eval_pp_{tf,fr}_s{3072,1234,42}.log`、`eval_pp_{struct2,cons2,consacc2}.log`;[physionpp §3.8](../../physion/physionpp_ood_longhorizon.md)。

## 3. 直训活路②:num_preds 越长、长程越好,无拐点（Fig 7）

by-horizon nMSE↓（epoch20 公平对比）:

| horizon | np8 | np20 | np20+sc | np28+sc |
|---|---|---|---|---|
| h16 | 0.016 | 0.022 | 0.012 | 0.008 |
| h32 | 0.140 | 0.033 | 0.024 | 0.010 |
| **h64** | 0.280 | 0.220 | 0.087 | **0.014**（1/19） |

num_preds 16→20→28 时 h64 单调降 0.136→0.087→0.014,**完全未见拐点**（真实动力学比合成域更吃长 rollout）。刚体 cos 0.96–0.99,布料形变是短板（deform_clothhit 0.610 / clothhang 0.367）。源:`physionpp/eval_pp_fr{,_np20,_np20sc,_np28sc}_e20*.log`;[§3.6](../../physion/physionpp_ood_longhorizon.md)。

## 4. 真 held-out scene OOD:泛化是分层的（⚠️单种子）

训练整场景排除,rollout cos（held-out vs full）:

| 场景（属性） | held-out | full | 判定 |
|---|---|---|---|
| mass_waterpush（质量） | **0.972** | 0.994 | 泛化成功（刚体质量可迁） |
| bouncy_wall（弹性） | 0.846 | 0.996 | 部分 |
| deform_clothhang（形变） | **0.263** | 0.982 | 崩（形变迁不动） |

**可迁移的是刚体表观动力学,形变迁不动。** GROUP 混合 nMSE=3.22 是 deform+静止分母 artifact,以 cos 分场景判（呼应陷阱1）。源:`physionpp/eval_np28sc_{ho_ood,full_seen}.log`;[§3.7](../../physion/physionpp_ood_longhorizon.md)。

## 5. init 消融:init 价值随目标数据规模/真实度递减（⚠️单种子）

physion 直训 h64 nMSE = **scratch 0.034 < cube(3D) 0.048 < pusht(2D) 0.058**。scratch 最好（数据足够,预训练 init 的域偏见反拖累）;cube(3D)>pusht(2D)（3D init 更近 physion）。**与 phyworld（pusht init > random）相反 → 数据越大越真,预训练 init 越没用。** 源:`physionpp/eval_pp_init_{scratch,cube,pusht}.log`;[§3.8③](../../physion/physionpp_ood_longhorizon.md)、[init_ablation_PLAN.md](../../physion/init_ablation_PLAN.md)。
