# OOD / 长程预测优化方案（自主规划）

**日期**：2026-07-05
**目标**：在 free-rollout 已默认的基础上，继续把 OOD 和长程预测提上来。
**约束**：写此文时 8 卡全被占（kejuan 3-7 + 用户自己的 cons/Physion 0-2），无空闲产能 → 优化按"GPU 空即自动launch"排队（[run_queue.sh](../run_queue.sh)），不等人。

---

## 0. 已知短板（来自 [FINAL_SUMMARY](FINAL_SUMMARY.md)）

| 短板 | 根因 | 对应优化 |
|---|---|---|
| collision 全面差（both-OOD 0.39） | 冲量非光滑，smooth accel MLP 扛不住 | **form-free consistency loss**（T1） |
| r/m-OOD 位置崩（ρ 0.29） | 编码器对外观(半径/质量)变化不鲁棒；无任何增广 | **外观/尺度增广**（T3） |
| 长程仍会缓慢漂 | 训练 rollout 只 8 步 | **更长训练 rollout num_preds↑**（T1） |
| 物理增益被稀释 | 2/192 slot 非承重；黑盒主导 | 承重(已做) / **extrinsic 化**（T3） |

---

## 1. 优化清单（按 价值×成本×把握 排序）

### Tier 1 — 高把握、低成本、已排队自动跑
1. **collision + consistency loss**（form-free 物理约束）：collision 是最大短板，consistency（约束预测速度=真实速度、无固定 accel 形式）正是为冲量设计。用户在 uniform 上测 cons，**collision 上没人测——这是最高价值的空白**。
2. **更长训练 rollout（num_preds 8→16）**：free-rollout 训得越长、部署长程越稳。uniform 已近天花板(h28 cos 0.97)，**collision 空间最大**（h28 cos 0.58）→ 优先 collision。
3. **承重+运动学 + 长 rollout 组合**：在 uniform/parabola 赢家配置上叠加 num_preds=16，看长程能否再进一步。

### Tier 2 — 中等成本，值得做
4. **rollout 课程（num_preds 随 epoch 2→8→16 增长）**：稳定训练 + 兼顾短长程。需小改 train.py。
5. **SWA / 末几 epoch ckpt 平均**：便宜的 post-hoc 泛化提升，无需重训。

### Tier 3 — 高成本高回报，需较大改动（暂不自动跑，列为路线）
6. **中心保持的尺度增广**（模拟半径 OOD，不破坏位置标签）：直接治 r/m-OOD 编码鲁棒性。当前 pipeline 无增广，需自定义 per-sequence 一致的尺度 jitter。
7. **PIWM extrinsic 化**：低维物理态做承重主 latent + 固定形式动力学 → 物理结构才能全面吃红利。
8. **离散/VQ latent**（PIWM 实测 OOD 最优的正则）。

---

## 2. 实际跑的 8 个（2026-07-05 20:29，打满 8 卡直跑）

> 起初写成"等空卡"队列（[run_queue.sh](../run_queue.sh)，判据=无计算进程），但用户指示"显存够就直接 co-locate 跑、别等独占"，故 GPU 3-7 空出后直接铺 8 个打满（含 co-locate 到 0/1/2 上，无 OOM）。collision 占 5 个（最大短板、最需 form-free）。

| GPU | run | 域 | 配置 | 验证 |
|---|---|---|---|---|
| 3 | col_structpos_cons1.0 | collision | 结构化+cons=1.0 | **form-free 治冲量** |
| 4 | col_structpos_cons0.3 | collision | cons=0.3 | 弱 cons |
| 5 | col_structpos_cons1.0+acc | collision | cons=1.0+2阶 | 加二阶一致性 |
| 6 | col_structdyn_areg_cons1.0 | collision | 可学accel+cons | cons+运动学组合 |
| 7 | col_baseline_fr_np16 | collision | 纯FR+num_preds16 | 长 rollout 救 collision |
| 1 | um_structcv_fr_pw100_np16 | uniform | 赢家+np16 | 长程再进一步 |
| 0 | par_structdyn_areg_fr_pw30_np16 | parabola | 赢家+np16 | 同上 |
| 2 | par_structpos_cons1.0+acc | parabola | cons+2阶 | form-free vs accel-MLP |

**判据**：collision cons 组 both-OOD < 纯 FR 0.393（form-free 是否治冲量）；np16 组长程 h28 较 np8 版提升。

监听器 `bifime6wd`，全完成自动叫醒我拉数、判优化、填 §3。

结果落地后接入 [FINAL_SUMMARY](FINAL_SUMMARY.md) / 本文 §3。

---

## 3. 结果（第一批 8 个，2026-07-05）

### 3.1 🎯 大发现：collision 靠"更长训练 rollout"大幅变好

| collision | ID | r/m | v | **both** | h28cos |
|---|---|---|---|---|---|
| baseline_fr (np8) | 0.379 | 0.183 | 0.609 | 0.393 | 0.581 |
| **baseline_fr np16** | **0.112** | **0.112** | **0.433** | **0.346** | **0.679** |

**num_preds 8→16 让 collision 全线暴涨**（ID 3倍、long-horizon h28 0.58→0.68）。冲量/反弹是多步现象，长 rollout 才学得到。**这是首个真正提升 collision 的手段。**

### 3.2 consistency loss 失败；num_preds 有域依赖

- **consistency loss 没治好 collision**：col cons 组 both-OOD 0.57-0.64，全部 **差于** 纯 FR 0.393；parabola cons(0.291) 也差于 accel-MLP(0.262)。form-free 假设未成立。
- **num_preds 有域依赖**：collision **大赢**，但 uniform(0.109→0.128)/parabola(0.262→0.329) **略差** → **简单域 np8 已够、别加长；复杂域(collision)吃长 rollout。**

### 3.3 结论 → 新优化原则

> **num_preds 应按动力学复杂度调**：collision 这种多步/冲量域调大（≥16），uniform/parabola 保持 8。这比"固定 num_preds=8 默认"更优。

### 3.4 collision num_preds 完整曲线（第二批，both-OOD↓ / h28cos↑）

| num_preds | ID | r/m | v | **both** | **h28cos** |
|---|---|---|---|---|---|
| 8 | 0.379 | 0.183 | 0.609 | 0.393 | 0.581 |
| 12 | 0.177 | 0.121 | 0.517 | 0.343 | 0.663 |
| 16 | 0.112 | 0.112 | 0.433 | 0.346 | 0.679 |
| 16(种子2) | 0.165 | 0.112 | 0.438 | 0.344 | 0.672 |
| **20** | 0.132 | 0.113 | 0.396 | **0.294** | **0.701** |
| 24 | 0.143 | 0.105 | 0.304 | 0.363 | 0.646 |

**甜点 = num_preds≈20**：both-OOD 0.393→**0.294（−25%）**、长程 h28 cos 0.58→**0.70**；单调升到 20，np24 回退（过长累积噪声）。**种子复现确认非 fluke**（np16: 0.346 vs 0.344）。

**其他确认**：
- 运动学在长 rollout 上仍不加分（col_structdyn_areg_np16 both=0.389 > 纯FR np16 0.346）→ collision 不吃运动学。
- 简单域 np16 有害：uniform 长程 cos 0.969→0.814、parabola both-OOD 0.313→0.416。

### 3.5 落定的优化原则

| 原则 | 内容 |
|---|---|
| **free-rollout 默认** | 主升力（已默认） |
| **num_preds 按域调** | collision 等复杂/冲量域 **≈20**；uniform/parabola 保持 **8**。是本轮最实用的新增益。 |
| 承重(pos_weight) + 运动学 | 仅光滑域（uniform/parabola）有效 |
| consistency loss | 未见收益，暂搁置 |
| freeze_encoder | 有害，别用 |

**collision 最优配方**：`free-rollout + num_preds≈20 + 纯 FR（不加 structured/运动学/pos_weight）`。

### 3.6 甜点精定 + pixel 复核（第三批，定稿）

**甜点 = num_preds 18-20**（latent both-OOD）：

| np | both-OOD | h28cos |
|---|---|---|
| 8 | 0.393 | 0.581 |
| 16 | 0.346 | 0.679 |
| **18** | **0.294** | 0.696 |
| **20** | **0.294** | **0.701** |
| 20(种子2) | 0.316 | 0.684 |
| 24 | 0.363 | 0.646 |

种子复现方向一致（0.294 vs 0.316，都远好于 np8）。**collision 提升由 latent 证据支撑，稳。**

**⚠️ collision 的 pixel 指标 decoder-limited、不可用（已定案）**：直接对比 np8 vs np20 pixel **几乎一样**（长程都 ~13-14dB、both-OOD 14.9 vs 15.3），而 latent 上 np20 明显更好（both 0.294 vs 0.393）→ pixel 抹平了 latent 差异。根因：**静态 decoder（REAL latent）也只有 17-23dB 且 ID<OOD 反常** → collision decoder 渲染能力本身就弱，**不是模型问题**。**collision 只能用 latent 尺；用 pixel 前须先修/重训 decoder。**（uniform pixel 正常可信。）np20 提升由 latent 证据支撑，成立。

### 3.7 collision 最终配方与后续
- **最优**：`free-rollout + num_preds≈18-20 + 纯 FR`，both-OOD 0.393→0.29（−25%）、长程 h28 cos 0.58→0.70。
- **待修**：collision universal decoder（欠训/分区标注）——修好才能用 pixel 复核。
- **未生效搁置**：运动学、consistency、pos_weight 在 collision 上均无正贡献。
