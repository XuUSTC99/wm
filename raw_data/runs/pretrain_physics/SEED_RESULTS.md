# Table 2 seed replication (auto-generated)

Generated: 2026-07-22 17:38 | complete: True

| domain | phys | s3072 | s1234 | s42 | mean±std | n |
|---|---|---|---|---|---|---|
| uniform | off | 0.1915 | 0.2301 | 0.2455 | **0.222±0.028** | 3 |
| uniform | on | 0.7500 | 0.5604 | 0.6615 | **0.657±0.095** | 3 |
| parabola | off | 0.3430 | 0.4196 | 0.6656 | **0.476±0.169** | 3 |
| parabola | on | 0.3747 | 0.4936 | 0.4727 | **0.447±0.063** | 3 |
| collision | off | 0.5378 | 0.5134 | 0.4535 | **0.502±0.043** | 3 |
| collision | on | 0.6351 | 0.6613 | 0.6488 | **0.648±0.013** | 3 |

## Δ (injection cost) per domain

| domain | Δ mean | 95% CI (Welch) | verdict |
|---|---|---|---|
| uniform | +0.435 | [+0.276, +0.594] | positive (injection hurts) |
| parabola | -0.029 | [-0.318, +0.260] | **overlaps zero — parity** |
| collision | +0.147 | [+0.074, +0.220] | positive (injection hurts) |
