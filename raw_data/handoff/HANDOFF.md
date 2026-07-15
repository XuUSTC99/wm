# CC Session 交接 —— 三条工作线的完整对话记录(2026-07-15 归档)

> **为什么有这个目录**:本项目的全部研究过程分散在 **3 个并行的 Claude Code session** 里(方法探索 / 评估基础设施 / 论文写作)。这些 session 的对话记录原本只存在本机 `~/.claude-chester/`,**随服务器停用一起消失**。此目录把三条线的**完整原始对话 jsonl** 收进 git,使**决策依据、失败路径、口径演变**在离开本机后仍可回溯。
>
> **3 个文件 / 36 MB / 11184 行对话**。数据源(log/eval json/config)见上级 [../README.md](../README.md) 与 [../MANIFEST.tsv](../MANIFEST.tsv)。

---

## 0. 三条线是怎么分工的(先读这个)

三个 session **职责严格分离**,互不重跑对方的实验——这是刻意的,避免重复劳动和口径打架:

| Session | 文件 | 职责 | 结论落在哪 |
|---|---|---|---|
| **lewm** | `session_lewm_a29d4510.jsonl` | **物理注入方法探索 + 机制诊断**(所有 phyworld 训练/消融) | `reports/6-24/`、`reports/7-11/{lbr_ablation,piwm_baseline}/` |
| **physion** | `session_physion_356c245c.jsonl` | **Physion/Physion++ 评估基础设施**(迁移评估、真实数据直训) | `reports/physion/` |
| **aaai_paper** | `session_aaai_paper_50905d5d.jsonl` | **论文故事线 + 证据体系**(汇总前两条线,不做新实验) | `reports/7-11/aaai_paper/` |

**⚠️ 接手须知**:要跑新的物理注入实验 → 走 lewm 线的口径;要动评估 pipeline → 走 physion 线;要改论文叙事/数字 → 走 aaai_paper 线,且**数字必须从 `01_results_ledger.md` 引、每个标注出处 log**。

---

## 1. 三个 session 详细

### 1.1 `session_lewm_a29d4510.jsonl` —— 方法探索线(最大,20 MB / 5300 行)

**时间**:2026-07-03 → 2026-07-15 | **首条**:"整理成报告吧"

**做了什么**:
- **七种物理注入 × 三域全扫描**:structpos(固定 slot)、probe(深监督)、运动学头(a=0/g/MLP)、consistency(form-free)、label-free(无标签先验)、grounded(有标签对照)、structposvel(位置+速度)。
- **LBR / pos_weight 全曲线消融**:权重 1→300(300× 范围),三域 × 四分区。
- **机制诊断**:梯度打架(加权 loss 比 15–125×)、intrinsic-dim 塌方(PR 41→4)、**probe-190 黑盒旁路实测**(三域一致)。
- **PIWM 官方移植**:extrinsic 架构外部 baseline,忠实移植到 phyworld。

**关键结论(已进 ledger)**:
- 21 格(7 注入 × 3 域)里 **20 格明确变差**,唯一"更低"的 probe·parabola 落在基线种子区间下沿 = 噪声。
- **LBR 只在 2/4 判决格回到持平**,uniform·r/m 与 collision 任何权重都救不回(collision 越加越差)→ 机制方向对但修不了根本。
- **probe-190**:钉了 slot 后,黑盒 190 维**仍独立冗余编码同一份位置**(ρ 0.79–0.92)→ 旁路是失效的架构性充分条件。
- **PIWM**:学到正确物理(parabola g_y≈真值)、ID/v-OOD 比 LeWM 准,但 **size/mass-OOD 崩**(ρ 0.33 vs 0.89)→ 物理结构买不到 OOD 鲁棒。

### 1.2 `session_physion_356c245c.jsonl` —— 评估基础设施线(7.4 MB / 3035 行)

**时间**:2026-07-04 → 2026-07-14 | **首条**:"我想 cc 只能我 junjxu 的 wm 这下面能用,其他人无法用,咋办"(开头是环境隔离,主体是 Physion)

**做了什么**:
- **Physion(原版 OCP)zero-shot 迁移评估**一键 pipeline(冻结 encoder + readout 分类)。
- **Physion++ 直训 + by-horizon 评估**(有 3D 位置/速度标注,可 rollout)。
- **增广实验**(appearance / scale)与 **num_preds 扫描**(8→28)。

**关键结论**:
- **zero-shot 迁移天花板 = random 架构先验 0.607**,所有训练配置都 <0.607(物理结构 pos_weight 0.551 最差、free-rollout 0.603 最接近)→ 别把"接近 random"当成"学到迁移能力"。
- **Physion++ FR/TF = 8.3×**(h64,三种子零重叠)——比合成域(2.2–3.6×)还大。
- **appearance 增广合成→真实反转 100×**(friction nMSE 0.062→6.44):真实场景里外观携带物理(摩擦/质量/材质)。
- **num_preds 28 + scale 增广:h64 nMSE 0.280→0.014(19×)**,单调无拐点。

**⚠️ 名词纠正(已全文订正)**:Physion / Physion++ 是 **TDW/Unity 照片级仿真**,**不是真实视频**。全库统一口径 "photorealistic-simulation",禁用 "real-world video"。

### 1.3 `session_aaai_paper_50905d5d.jsonl` —— 论文故事线(8.3 MB / 2849 行,本次归档时仍在进行)

**时间**:2026-07-06 → 2026-07-15 | **首条**:"承重编码(pos_weight) 换个表述词,完全看不懂"

**做了什么**:
- 把前两条线的数据汇总成 **`01_results_ledger.md`(C1–C7 主张台账)**。
- 写 **`06_storyline.md`**(9 步逻辑链)+ **`detail/`** 论据文件(每个带 🎯 一句话结论)+ 9 张图 + LaTeX 初稿。
- **多轮诚实性审计**(本线最大价值,见下)。

**关键产出**:`reports/7-11/aaai_paper/`——`01_results_ledger.md`(数字总账)、`06_storyline.md`(主线)、`02_story_and_novelty.md`(名词表+审稿预案)、`detail/*.md`(逐条论据)、`figures/storyline_figures.py`(一键重画,数字内嵌+源注释)。

**⚠️ 这条线纠正过的错误(接手别再犯)**:

| 原表述 | 问题 | 订正 |
|---|---|---|
| "复现 deep-sup(2504.03861)见其提升翻负" | **对该论文的 mischaracterization**——它主报告用的是**可信指标 pred_loss**(不是 probe-ρ 对偶量),结论**正确** | 改为:deep-sup 在**低维状态 latent**(8-D、probe 占 38%)成功是对的;搬到**高维视觉 latent**(192-D、占 2%)失效属**塌方/稀释机制**,是域边界、非指标问题 |
| "real-world video benchmarks" | 事实错误(Physion 是仿真) | "photorealistic-simulation" |
| "承重 / load-bearing"(当机制解释用) | 自造黑话,不通用 | 机制处改说 "slot 在 loss 里占比极低 + 被黑盒旁路绕过";术语锚 *not load-bearing* 仅在核心概念处保留 |
| "物理结构只在既承重、又匹配该域动力学时才帮" | 两个词都在打哑谜,且"才帮"像"找到了有用条件" | 拆成两条大白话:① slot 占比高到不被旁路绕过 ② 编的量在该域**可外推**(parabola 速度线性✓/uniform 常数冗余/collision 跳变✗);且明说这是**机制签名、非通用方法** |
| LBR "最小修复 / 净超基线" | over-claim(单种子噪声) | "只 2/4 格回持平、无净增益",框架改为**机制的可证伪验证** |

---

## 2. 怎么用这些 jsonl

**格式**:Claude Code 原始对话记录,**每行一个 JSON**(含 `type`/`timestamp`/`message`,以及完整的 tool call 与 tool result)。

```bash
# 快速看某 session 的所有用户提问(还原意图脉络)
python3 -c "
import json
for line in open('session_lewm_a29d4510.jsonl', errors='ignore'):
    try:
        d = json.loads(line)
        if d.get('type') == 'user':
            c = d.get('message', {}).get('content')
            t = c if isinstance(c, str) else ' '.join(x.get('text','') for x in c if isinstance(x, dict))
            if t.strip() and not t.startswith('<'):
                print(f\"[{d.get('timestamp','')[:16]}] {t.strip()[:120]}\")
    except: pass
"

# 搜某个结论是怎么来的(如 probe-190)
grep -o '.\{200\}probe-190.\{200\}' session_lewm_a29d4510.jsonl | head
```

**在本机恢复上下文**(session 还在时):`claude --resume <session-id>`,id 即文件名里的 uuid。
**离开本机后**:jsonl 即完整记录,可直接作为新 session 的上下文材料喂入,或用上面脚本检索。

---

## 3. 关键结论速查(不想读 11184 行就看这个)

**主线一句话**:物理信息"存在"于 latent(可解码 ρ 0.80–0.96),但预测**不依赖你注入的那一份**——黑盒早已冗余编码同一份位置,预测绕过注入的 slot 走黑盒(probe-190 实证)。于是往共享 latent 嫁接物理**状态**只是塞冗余(不提升、反占容量与梯度)。真正让模型遵守物理的是**训练协议**(free-rollout),不是结构先验。

| 主张 | 数字 | 出处 |
|---|---|---|
| free-rollout 是唯一跨域主升力 | 合成 2.2–3.6×、Physion++ h64 **8.3×**(均三种子) | `detail/free_rollout_evidence.md` |
| 物理结构系统性有害 | 21 格 20 格变差;from-scratch 更狠(Δ+0.558) | `detail/why_physics_structure_fails.md` |
| 机制 = 占比低 + 黑盒旁路 | 2/192 维、梯度比 15–125×、PR 塌 39–90%、probe-190 ρ 0.79–0.92 | 同上 |
| LBR 只回持平 | 2/4 格,collision 越加越差 | `detail/load_bearing_reweighting.md` |
| 评测陷阱 | cos 升 1.50× 而 nMSE 崩到 0.21;迁移封顶 random 0.607 | `detail/evaluation_traps.md` |
| 增广合成→仿真反转 | friction nMSE 100× | `detail/augmentation_synthetic_vs_real.md` |

完整台账见 [`reports/7-11/aaai_paper/01_results_ledger.md`](../../reports/7-11/aaai_paper/01_results_ledger.md);故事线见 [`06_storyline.md`](../../reports/7-11/aaai_paper/06_storyline.md)。

---

## 4. 接手最该知道的三个坑

1. **种子命名不一致**(最容易踩):3072 是默认种子、**无 `_s` 后缀**且在 `runs/structdyn_eval/`;1234/42 带 `_s` 后缀在 `runs/aaai_p0/`。**不存在 `rollout_*_baseline_fr_s3072.log`**(phyworld 三域)。详见 [../README.md §0](../README.md)。
2. **parabola 的 nMSE 除零爆点**:both-OOD 在 h28 附近球飞出框 → 目标方差→0 → nMSE 飙百万。**parabola 判决一律走 r/m-OOD 分区**。
3. **cos/probe-ρ 是训练目标的对偶量**,加对应 loss 必涨——**判决只能用 nMSE/pixel**,cos/probe 仅作诊断;且 probe-ρ **分区间不可直接比**(ID 被 range restriction 系统性压低)。
