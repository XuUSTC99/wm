# 跨模型泛化实验：第二个 JEPA 家族模型（AAAI 冲刺，2026-07-16）

**目的**：把论文两大发现从 LeWM 单模型推广到 JEPA 家族——堵"你只测了一个自研模型"这个最大审稿质疑。

## 模型选型：DINO-WM 风格（冻结 DINOv2-small + 同一套 AR predictor）

**为什么选它而不是 V-JEPA 2**（都考虑过）：
| 维度 | DINO-WM 风格（选定） | V-JEPA 2-AC |
|---|---|---|
| JEPA 家族身份 | ✅ 已发表（DINO-WM: frozen DINOv2 + latent predictor；V-JEPA2-AC 同为 frozen encoder + AR predictor 设计） | ✅ 旗舰 |
| 特征粒度 | **per-frame**（和现有 frameskip/action 管线零冲突） | 视频 tubelet(2帧)，对齐 frameskip/action 需重做数据管线 |
| 规模/速度 | 22M，冻结，**35min/run、6.3GB** → 27 run 全并行 | ViT-L 300M+，工程+算力 10 天赌不起 |
| 与 LeWM 的差异度 | 换 backbone 出身(DINOv2 SSL vs pusht JEPA)、冻结 vs 可训、384 vs 192 维 | 更大差异但不可控 |

**实现**（le-wm/train.py `+encoder_type=dinov2 +freeze_encoder=true`）：encoder 换成冻结 `facebook/dinov2-small`（384-d CLS），已有的可训练 projector(384→192) 自动成为 adapter，**下游全部不动**（同一 ARPredictor/losses/FR-TF 开关/eval）→ 严格受控的 cross-backbone 消融。

**前提验证（冒烟已过）**：冻结 DINOv2 CLS 的位置线性可解码 REAL-emb ρ≈0.93 → "latent 已编码状态"前提在第二个 backbone 成立，注入的是"已有的状态"。

## 实验矩阵（27 runs，8 卡全并行，~2h）

- **发现二 headline（FR≫TF）**：3 域 × {TF(np1,b128), FR(np8)} × 3 种子 = 18
- **发现一（注入失效）**：3 域 × {structpos_pw30(slot), probeF2(深监督), cons(一致性)} = 9，臂配置逐字复刻 LeWM（run_fillcells_queue.sh 的 PROBE/SLOT 定义）
- 判决口径与 LeWM 一致：nMSE↓，uniform/collision 用 both-OOD、parabola 用 r/m-OOD（爆点规则）

## 预期写法
- 若复现（FR 大幅赢 + 注入不赢）→ "两个架构差异显著的 JEPA 实例上结论一致" → 标题的"JEPA 普遍性"成立
- 若部分不复现 → 诚实写差异 + 归因（冻结 encoder 时注入只能走 adapter，机制上更接近"外挂"）

*队列*：`run_dinowm_queue.sh`；*log*：`/data1/likun-share/junjxu/runs/dinowm/{train,rollout}_dinowm_*.log`
