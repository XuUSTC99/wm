# 老 R² probes（已弃用）

5-26 report 把评估指标从 R² 换成了 **MSE + Pearson ρ + 2-layer MLP**（对齐 LeWM Table 1）。
这些老 probe 脚本用 R² 报告，**已不在主流程中使用**，归档备查。

## 不要再用的脚本（被 fullfit 系列替代）
- `probe_ood.py`, `probe_ood_per_partition.py`, `probe_ood_uniform.py` — K=1 + ID-only fit，**5-26 report 推翻的初版协议**
- `probe_multiframe.py` — K=4 R² 单文件实验
- `probe_all_targets.py`, `probe_collision_encoder.py`, `probe_lewm_encoder.py`, `probe_lewm_pusht_only.py`, `probe_dit_zeroshot.py` — 早期 encoder-specific R² probes

## 当前应该用的脚本（保留在 scripts/）
- `probe_mlp_mse_pearson.py` — **正版 LeWM-aligned MLP probe**（MSE + Pearson ρ + nMSE）
- `probe_ood_fullfit.py` — Ridge K=4 mixed-fit per-partition（collision）
- `probe_ood_uniform_fullfit.py` — Ridge 同款（uniform_motion / parabola）
- `convert_to_lewm.py` — phyworld hdf5 → LeWM h5 转换
