# PIWM Deep-Supervision — uniform_motion + collision（follow parabola）

**生成时间**：2026-06-01 18:32（由 run_piwm.sh 自动汇总）

协议同 [5-26/piwm_deepsup_results.md](../5-26/piwm_deepsup_results.md)：ID-only 1k 训练，probe on FT loss，rollout 评估在全 OOD eval 集。4 臂 = baseline(无probe) / pos-only(训练单帧) / pos+vel(训练单帧) / mf4(训练多帧 frames=4)。velocity 监督列：uniform=action(速度), collision=state(速度)。

> ⚠️ 所有解码列用推理 K=4；'单帧/多帧'指训练时 probe 吃几帧。


## uniform_motion

### vx 解码 ρ（K=4）

| partition | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| ID | +0.440 | +0.804 | +0.596 | +0.642 |
| r/m-OOD | +0.564 | +0.737 | +0.700 | +0.661 |
| v-OOD | +0.782 | +0.944 | +0.924 | +0.942 |
| both-OOD | +0.893 | +0.928 | +0.902 | +0.886 |

### vy 解码 ρ（K=4）

| partition | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| ID | +nan | +nan | +nan | +nan |
| r/m-OOD | +nan | +nan | +nan | +nan |
| v-OOD | +nan | +nan | +nan | +nan |
| both-OOD | +nan | +nan | +nan | +nan |

### latent cos by partition

| partition | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| ID | +0.9647 | +0.9669 | +0.8586 | +0.8668 |
| r/m-OOD | +0.7642 | +0.8846 | +0.7100 | +0.6721 |
| v-OOD | +0.9242 | +0.9406 | +0.8785 | +0.8951 |
| both-OOD | +0.8597 | +0.8703 | +0.7959 | +0.7665 |

### latent cos by horizon

| h | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| 1 | +0.9924 | +0.9940 | +0.9848 | +0.9880 |
| 2 | +0.9841 | +0.9855 | +0.9666 | +0.9713 |
| 4 | +0.9636 | +0.9684 | +0.9277 | +0.9215 |
| 8 | +0.9119 | +0.9316 | +0.8567 | +0.8115 |
| 16 | +0.8445 | +0.8653 | +0.7541 | +0.7061 |
| 28 | +0.8432 | +0.8816 | +0.8062 | +0.7997 |

## collision

### vx 解码 ρ（K=4）

| partition | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| ID | +0.241 | +0.602 | +0.657 | +0.722 |
| r/m-OOD | +0.327 | +0.599 | +0.463 | +0.573 |
| v-OOD | +0.091 | +0.402 | +0.389 | +0.351 |
| both-OOD | +0.272 | +0.601 | +0.419 | +0.413 |

### vy 解码 ρ（K=4）

| partition | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| ID | +0.168 | +0.460 | +0.573 | +0.727 |
| r/m-OOD | +0.366 | +0.593 | +0.485 | +0.591 |
| v-OOD | +0.464 | +0.490 | +0.293 | +0.514 |
| both-OOD | +0.683 | +0.594 | +0.521 | +0.514 |

### latent cos by partition

| partition | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| ID | +0.5071 | +0.6156 | +0.5886 | +0.6232 |
| r/m-OOD | +0.4984 | +0.5748 | +0.5332 | +0.5213 |
| v-OOD | +0.3613 | +0.4582 | +0.3389 | +0.4008 |
| both-OOD | +0.4732 | +0.4830 | +0.4562 | +0.4384 |

### latent cos by horizon

| h | baseline | pos-only(训练单帧) | pos+vel(训练单帧) | mf4(训练多帧) |
|---|---|---|---|---|
| 1 | +0.9905 | +0.9916 | +0.9880 | +0.9905 |
| 2 | +0.9672 | +0.9726 | +0.9634 | +0.9675 |
| 4 | +0.8631 | +0.8816 | +0.8577 | +0.8723 |
| 8 | +0.6290 | +0.6635 | +0.6311 | +0.6547 |
| 16 | +0.3404 | +0.4227 | +0.3755 | +0.3488 |
| 28 | +0.2357 | +0.2596 | +0.1979 | +0.1978 |

---

## 关键问题（待人工解读）

1. parabola 上的结论是否复现：单帧 probe 砸 vx 高速 OOD？mf4 是否修回 + 长程 cos 最佳？
2. collision（action=加速度、2 球）与 uniform（vx 恒等）是否表现一致？
3. baseline 的 K=4 vx 在高速 OOD 是否仍是'位置差分白嫖'的强基线？

日志：`reports/6-2/logs/`（train_*.log / rollout_*.log / orchestrator.log）