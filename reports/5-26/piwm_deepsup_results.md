# 实验 A：PIWM-style Deep-Supervision Linear Probe

**日期**：2026-05-31
**测试域**：⚠️ **仅 parabola（抛物线，自由落体 + 重力）一个域**。collision / uniform_motion 尚未做（§8 待办）。所有数字都只针对 parabola。
**动机**：[rollout_results.md](rollout_results.md) 暴露两个问题——(1) AR rollout 的 latent cosine 随 horizon 衰减（长程漂移），(2) §4 从 rollout latent 解码 pos/vel 偏弱。本实验借鉴 PIWM 思路（arXiv:2412.12870 + deep-supervision arXiv:2504.03861），在 LeWM FT 时加一项 **linear probe 监督 loss**，看能否同时改善这两点。
**脚本**：训练 [le-wm/train.py](../../le-wm/train.py)（加 `loss.probe`）；评估 [rollout_eval_id1k.py](../../phyworld/scripts/rollout_eval_id1k.py)

---

## 1. 方法

在 LeWM FT 的 loss 里加一项 deep-supervision linear probe（默认关，`loss.probe.enabled` 开关，方便消融）：

```python
# train.py  lejepa_forward
probe_pred = self.model.probe_head(emb)              # Linear(192 → proprio_dim)
probe_loss = (probe_pred - proprio_normalized)**2    # 监督 position
loss = pred_loss + λ_sigreg·sigreg + λ_probe·probe_loss
```

- `probe_head` = **单层 Linear(192 → 2)**，对齐 deep-supervision 论文（不是 PIWM 的 MLP extractor）
- 监督目标 = **proprio（position，z-score 归一化）**，对齐 PIWM intrinsic 的 `state[:,:2]`
- 在 **projector 空间**做（predictor 的工作空间），所以 rollout 预测的 latent 也受益
- `λ_probe = 1.0`

**配置**（跟 baseline 完全一致，只多 probe 项）：
- data = `phyworld_parabola_id1k.h5`（PhyWorld 官方 ID-only 1000 trajs）
- init = pusht weights, 20 epoch, GPU 3
- baseline 臂 = `parabola_paperinit_id1k`（`loss.probe.enabled=false`）

**训练有效性验证**：`probe_loss` 从 **1.475 → 0.036**（降 40×），`pred_loss` 0.0046（无退化）。说明 emb 被成功逼成"线性可读 position"，且没破坏原始预测任务。

---

## 2. 结果：latent cosine vs rollout horizon（长程漂移）

aggregate over test trajs：

| horizon | baseline | +probe | Δ |
|---|---|---|---|
| h=1 | 0.978 | 0.984 | +0.006 |
| h=2 | 0.932 | 0.958 | +0.026 |
| h=4 | 0.864 | 0.917 | **+0.053** |
| h=8 | 0.814 | 0.877 | **+0.063** |
| h=16 | 0.535 | 0.633 | **+0.098** |
| h=28 | —† | —† | 数值发散尾，不可比 |

† 两臂 h=28 都有少数 traj 的 AR rollout 数值发散（nMSE 爆炸到 1e7~1e8），cosine 不稳，略去。

→ **中长程 horizon 漂移被显著压住**（h=8 +0.063, h=16 +0.098）。

---

## 3. 结果：per-partition latent cosine（ID→OOD，最强信号）

aggregate over all horizons，test trajs：

| partition | baseline | +probe | Δ |
|---|---|---|---|
| ID | 0.935 | 0.978 | +0.043 |
| r-OOD | 0.741 | 0.886 | **+0.145** 🔥 |
| v-OOD | 0.581 | 0.701 | **+0.120** 🔥 |
| both-OOD | 0.511 | 0.658 | **+0.147** 🔥 |

→ **OOD 上的提升比 ID 还大**（+0.12~0.15）。deep-supervision 让 rollout 在没见过的 OOD partition 上漂移慢得多——这是本实验最强的结果，说明把 position 钉进 emb 让 predictor 的外推更稳健。

---

## 4. 结果：从 rollout latent 解码 position（§4 旧弱项）

probe 应用到 **predicted（rollout）emb**，per-partition：

| partition | baseline pos_x | +probe pos_x | baseline pos_y | +probe pos_y |
|---|---|---|---|---|
| ID | 0.633 | **0.948** 🔥 | 0.667 | **0.978** 🔥 |
| r-OOD | 0.756 | 0.892 | 0.794 | 0.967 |
| v-OOD | 0.634 | 0.853 | 0.803 | 0.907 |
| both-OOD | 0.646 | 0.806 | 0.760 | 0.911 |

→ **position 解码大幅改善**（+0.15~0.32），ID position 从 ~0.63 直接到 ~0.96。因为 probe loss 监督的就是 position，emb 高度线性可读，rollout 后仍保持。这正面解决了 rollout_results §4 "解码弱" 的问题。

---

## 5. velocity：vy 被救起，vx 在高速 OOD 上反被 probe 牺牲

**K=1（单帧）解码 velocity 几乎失败**——单帧本就读不出速度（§6 早证 vel 需多帧差分）。

**修法：K=4——把 rollout 预测出的连续 4 个 latent 拼起来再解码**（`[ê_{k-3},…,ê_k]`，跟 §6 在真实 emb 上的 K=4 同款，只是作用在预测 latent 上）。澄清：rollout **能**做 K=4，predictor 吐的是整条预测序列，拼 4 帧即可；唯一真约束是必须留在 projector 空间（projector 不可逆）。

### 5.1 vy（垂直，重力驱动）—— deep-sup + K=4 都帮忙

从 rollout 预测 latent 解码 **vy ρ**（per-partition）：

| partition | baseline K=1 | baseline K=4 | +probe K=1 | +probe K=4 |
|---|---|---|---|---|
| ID | 0.734 | **0.870** | 0.983 | **0.987** |
| r-OOD | 0.873 | 0.834 | 0.978 | 0.979 |
| v-OOD | **0.413** | **0.873** 🔥 | 0.912 | 0.923 |
| both-OOD | 0.603 | **0.794** | 0.903 | 0.935 |

→ vy 上 K=4 和 +probe **都正向**：baseline v-OOD K=1→K=4 从 0.41→0.87；+probe 进一步抬到 0.92。原因：重力让 vy 跟 pos_y 强耦合，监督 position 顺带帮了 vy。

### 5.2 vx（水平，常量速度）—— +probe 在高速 OOD 上**反而更差**

从 rollout 预测 latent 解码 **vx ρ**（per-partition）：

| partition | baseline K=1 | +probe K=1 | baseline K=4 | +probe K=4 |
|---|---|---|---|---|
| ID | 0.328 | 0.287 | 0.304 | **0.585** ✅ |
| r-OOD | 0.407 | 0.462 | 0.612 | 0.662 ✅ |
| v-OOD | **0.584** | 0.413 ❌ | **0.649** | 0.450 ❌ |
| both-OOD | 0.527 | 0.603 | **0.696** | 0.515 ❌ |

→ **这是 +probe 的一个真实代价，不是噪声**：
- ID / r-OOD 上 +probe 帮了 vx（K=4: ID 0.30→0.59, r-OOD 0.61→0.66）
- **但高速 partition（v-OOD / both-OOD）上 +probe 把 vx 拉低 0.18~0.20**（K=4: v-OOD 0.649→0.450, both-OOD 0.696→0.515），K=1 上同样（v-OOD 0.584→0.413）

**机制解读**：probe **只监督 position，且只在 ID 数据（vx∈[1,4]）上**。把 emb 压成 "position 线性可读" 的重组对 vy 有利、对解耦的 vx 不利——尤其外推到没见过的高速 OOD，vx 这种细微速度区分被挤压。**监督 position ≠ 监督 velocity，对 vx 甚至是负迁移**。

### 5.3 试图修复：训练时直接监督 velocity（pos+vel）—— **失败，反证单帧极限**

把 `loss.probe.target` 扩成 `[proprio, action]`（probe_head = Linear(192→4)，同时监督 pos+vel），重训 parabola 20ep（probe_loss→0.16, pred_loss 0.0087 无退化）。**预期它能修掉 vx 负迁移。结果:没修好,反而更糟。**

**三臂 vx ρ 对比（K=4，从 rollout 预测 latent 解码）：**

| partition | baseline | pos-only probe | **pos+vel probe** |
|---|---|---|---|
| ID | 0.304 | 0.585 | **0.612** ✅ |
| r-OOD | **0.612** | 0.662 | 0.480 ❌ |
| v-OOD | **0.649** | 0.450 | 0.483（仍 < baseline ❌）|
| both-OOD | **0.696** | 0.515 | 0.612（仍 < baseline ❌）|

**三臂 vx ρ 对比（K=1）**：pos+vel 的 K=1 vx 甚至比 pos-only 更低（both-OOD 0.603→0.363, r-OOD 0.462→0.282），且 position 略降（r-OOD pos_x 0.892→0.880）。

**关键洞察（负结果，有价值）**：

> 直接监督 velocity **救不了 vx,因为 probe 作用在单帧 emb_t 上,而单帧图像物理上不含瞬时速度**（尤其 vx 是水平常量速度,看一帧球的位置完全推不出它多快）。监督 velocity 时 probe loss 把 emb 往一个**不可能完成的任务**上拽,梯度矛盾/噪声 → 学不到 vx,反而**干扰了 position 对齐**(K=1 vx、position 双降)。
>
> 这反证了 **PIWM 原则 1（对齐单帧 latent 到物理量）对"速度类、需要时序的量"天生失效**。速度本质需要跨帧信息,单帧 deep-supervision 改变不了这个物理事实。

**补充：latent cosine 与 vx Pearson 背离**。pos+vel 虽然 vx 线性可读性失败,但 **predictor 输出向量的整体 cosine 反而在高速 OOD 上最好**（v-OOD 0.771, both-OOD 0.717,见 §5.5）。说明 vx 信息可能没丢、只是被编码得**非线性不可读**——"失败"仅限于线性 probe。

### 5.4 正解：训练时**多帧**监督（mf4，probe 吃 4 帧窗口）—— 成功

把 probe 改成吃 K=4 帧窗口（`loss.probe.frames=4`，probe_head = Linear(4×192=768 → 4)，预测窗口最后一帧的 pos+vel）。这样 probe 能**跨帧差分恢复速度**（物理上可行的任务），且不再强迫每个单帧 position-刚性。probe_loss→0.047（远低于单帧 pos+vel 的 0.16，因为多帧真能解出速度）。

**四臂 vx ρ 对比 —— ⚠️ 全部用推理时 K=4 解码;列只在「训练时监督方式」不同：**

| partition | baseline<br>(无 probe) | pos-only<br>(训练单帧) | pos+vel<br>(训练单帧) | **mf4<br>(训练多帧)** |
|---|---|---|---|---|
| ID | 0.304 | 0.585 | 0.612 | **0.702** 🔥 |
| r-OOD | 0.612 | 0.662 | 0.480 | **0.643** ✅ |
| v-OOD | **0.649** | 0.450 | 0.483 | 0.518 ⬇ 仍低 baseline |
| both-OOD | **0.696** | 0.515 | 0.612 | 0.659（≈ baseline）|

（"单帧/多帧" = **训练时 probe 吃几帧**;解码一律 K=4。inference-多帧 ≠ training-多帧,别混。）

**四臂 latent cos by horizon（同样全推理 K=4 之外的指标，纯 rollout 漂移）：**

| horizon | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | **mf4(训练多帧)** |
|---|---|---|---|---|
| h=4 | 0.864 | 0.917 | 0.867 | **0.925** |
| h=8 | 0.814 | 0.877 | 0.846 | **0.875** |
| h=16 | 0.535 | 0.633 | 0.648 | **0.702** 🔥 |

→ **mf4 是综合最佳**：把单帧版砸掉的 vx 基本拉回（ID 0.70 四臂最高、r-OOD 回到 baseline、both-OOD≈baseline），同时**长程 cos 全 horizon 第一**（h=16 0.702 vs baseline 0.535），position/vy 也保持最好。

**为什么连 mf4 在 v-OOD vx 上仍低于 baseline（0.52 vs 0.65）—— 不是 bug，是 ID-only 监督的固有上限**：

> baseline 的 vx-on-OOD 是**白嫖的几何操作**：它没 probe loss，encoder 只忠实编码**位置**（几何量，任何速度都编得准），K=4 解码 = 4 帧位置差分 ≈ 速度，**位置差分在任何速度下都成立** → v-OOD 高速球也拿 0.65。
>
> 而 probe loss（无论单/多帧）**只在 ID 速度范围（vx∈[1,4]）上训**，把 emb 重组成"ID 速度线性可读"，这个重组**为 ID 调好、外推到 v-OOD（更高速）不如 baseline 那个未被动过的纯位置编码**。单帧 probe 还把位置编码也搞刚性（v-OOD 砸到 0.45）；多帧不强迫单帧刚性，恢复了大部分位置差分能力（回到 0.52），但**没法在最极端的未见速度上反超 baseline 的几何外推**。
>
> 推论：v-OOD vx 这点残差**不该靠"更强 probe"修**——它是"只在 ID 监督"的固有天花板。要在 v-OOD vx 反超 baseline，要么把 OOD 速度放进训练（就不是 ID→OOD 测试了），要么上原则 2（速度作为受方程约束的状态变量，积分在任何速度都对）。

### 5.5 结论

1. **vx 负迁移的根因 = 训练单帧监督强迫每帧 position-刚性**，挤掉了 OOD 速度的细微结构。**改成训练多帧窗口监督即可基本修复**（mf4 把 vx 从单帧版的 0.45~0.52 拉回 0.52~0.70）。
2. **训练多帧监督 = 真正的正解**：同时拿到 deep-sup 的长程 cos / position / vy 好处，又不砸 vx。验证了用户的判断"多帧 probe 才对"。注意区分:这里指**训练时** probe 吃多帧;**推理时**所有臂都用 K=4 解码。
3. **单帧 deep-supervision 对需要时序的量（velocity）天生失效**——单帧图像不含瞬时速度，直接监督只会注入噪声（pos+vel 单帧版 K=1 vx、position 双降）。
4. **v-OOD vx 仍低于 baseline 是 ID-only 监督的固有上限**，非 bug：baseline 靠"位置差分"这个免训练几何操作，对任何速度都外推得好;任何在 ID 上训的 probe 都为 ID 速度调过、外推到极端高速略亏。要反超得放 OOD 速度进训练（破坏 ID→OOD 设定）或上原则 2。
5. **两个指标要一起看**：vx Pearson（线性可读性）和 latent cosine（整体保真）会背离——pos+vel 的 vx ρ 失败但 cosine 高，说明信息在、只是非线性。mf4 两个指标都好，是最干净的方案。
6. **caveat**：所有方案长 horizon（h≥16）都仍受 AR 漂移限制（cosine 跌、h=28 数值发散）；根治长程要上 PIWM 原则 2（physics-structured dynamics）或多步 rollout loss 重训。

---

## 6. 小结

四臂在 parabola 上（"单帧/多帧" = 训练时 probe 吃几帧；推理一律 K=4 解码）：

| 指标 | baseline<br>(无 probe) | 训练单帧 probe | **训练多帧 probe (mf4)** |
|---|---|---|---|
| 长程 cos h=16 | 0.535 | 0.63~0.65 | **0.702**（最佳）|
| position 解码 | 中 | ✅ ID→0.96 | ✅ 保持 |
| vy 解码 | 中 | ✅ 提升 | ✅ 保持 |
| vx 高速 OOD（K=4）| 0.65/0.70（好）| ❌ 砸到 0.45/0.52 | ✅ 拉回 0.52/0.66 |
| pred_loss | — | 无退化 | 无退化（0.008）|

> **结论**：PIWM-style deep-supervision 在 parabola 上能改善长程 rollout cos、position、vy。**单帧 probe 的代价是砸了 vx（高速 OOD 负迁移 −0.18~0.20），且直接监督 velocity 也救不回（单帧读不出速度）**。**正解是多帧（K=4）监督**（§5.4）：probe 吃 4 帧窗口,跨帧差分恢复速度、不再过度约束单帧 → **同时拿到全部 deep-sup 好处 + 把 vx 基本修回 + 长程 cos 四臂最佳**。这验证了 PIWM 原则 1 在"监督方式正确(多帧)"时有效；想根治长 horizon 漂移仍需原则 2（physics-structured dynamics）。

---

## 7. 复现 & 数据

| 项 | 路径 |
|---|---|
| 训练（+probe）| `cd ~/lewm_run && CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u train.py data=phyworld_parabola_id1k loss.probe.enabled=true loss.probe.weight=1.0 loss.probe.target=proprio output_model_name=lewm_parabola_piwm_probe_id1k subdir=parabola_piwm_probe_id1k trainer.max_epochs=20 +init_from_ckpt=.../lewm_paper_pusht/weights.pt` |
| +probe ckpt | `~/.stable_worldmodel/parabola_piwm_probe_id1k/lewm_parabola_piwm_probe_id1k_epoch_20_object.ckpt` |
| baseline ckpt | `~/.stable_worldmodel/parabola_paperinit_id1k/...epoch_20...` |
| 配置开关 | `le-wm/config/train/lewm.yaml` → `loss.probe.{enabled,weight,target}`（默认 enabled=false；target 可为单列或列表 `[proprio,action]`）|
| **pos+vel** 训练 | `... loss.probe.enabled=true 'loss.probe.target=[proprio,action]' output_model_name=lewm_parabola_piwm_posvel_id1k subdir=parabola_piwm_posvel_id1k ...` |
| **pos+vel** ckpt | `~/.stable_worldmodel/parabola_piwm_posvel_id1k/...epoch_20...` |
| +probe ckpt | `~/.stable_worldmodel/parabola_piwm_probe_id1k/lewm_parabola_piwm_probe_id1k_epoch_20_object.ckpt` |
| baseline ckpt | `~/.stable_worldmodel/parabola_paperinit_id1k/...epoch_20...` |
| **mf4 多帧** 训练 | `... loss.probe.enabled=true 'loss.probe.target=[proprio,action]' loss.probe.frames=4 output_model_name=lewm_parabola_piwm_mf4_id1k subdir=parabola_piwm_mf4_id1k ...` |
| **mf4 多帧** ckpt | `~/.stable_worldmodel/parabola_piwm_mf4_id1k/...epoch_20...` |
| rollout log（4 臂）| /tmp/rollout_parabola.log · _piwmprobe.log · _posvel.log · _mf4.log |
| K=4 对比 log（baseline vs pos-only）| /tmp/rollout_k4_cmp.log |
| 训练 log | /tmp/lewm_parabola_piwm_{probe,posvel,mf4}_train.log |
| 配置实现 | `loss.probe.frames`（1=单帧, K=多帧窗口）；[train.py](../../le-wm/train.py) probe block + probe_head 构建；K=4 eval decode 在 [rollout_eval_id1k.py](../../phyworld/scripts/rollout_eval_id1k.py)（`--ckpt`/`--tag` 切模型）|

注：路径前缀 `agent_memory` 已重命名为 `am`（`/home/qlib/am/wm/...`）；rollout 脚本内路径已同步更新。

## 8. 待办

- [x] velocity 训练时**单帧**监督版（`target=[proprio,action]`）→ **失败**（§5.3）。单帧读不出速度,救不回 vx。
- [x] **多帧 probe**（`frames=4`）→ **成功**（§5.4）。vx 基本修回 + 长程 cos 四臂最佳。**这是本实验的正解。**
- [ ] **PIWM 原则 2**：把 ARPredictor 换成 physics-structured dynamics（速度作为受方程约束的状态变量）——根治长 horizon 漂移
- [ ] mf4 v-OOD 的 vx 仍略低 baseline（0.52 vs 0.65）：试更大 K 或 frames 也用于 position
- [ ] 推广到 collision / uniform_motion 两域
- [ ] 正式消融：同一域跑 `enabled=false` vs `true`（当前 baseline 用更早的 `parabola_paperinit_id1k`，严格消融应同 commit 重跑 enabled=false）
- [ ] λ_probe sweep（0.1 / 1.0 / 10.0）+ frames sweep（2 / 4 / 全窗）
