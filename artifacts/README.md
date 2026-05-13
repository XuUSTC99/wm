# artifacts/

Intermediate files produced by experiments. Layout:

| Subdir | Tracked in git? | Contents |
|---|---|---|
| `logs/` | ✅ yes | Run stdout logs — kept as evidence backing the reports under `../reports/` |
| `embeddings/` | ❌ no (large, regenerable) | Cached encoder embeddings (e.g. `dit_xl_collision_emb_32k.npy` ~147 MB) |
| `scripts/` | ✅ yes | One-off helper scripts (data subsetting, conversion utilities) |

## Index

### Logs

- [`logs/dit_xl_zeroshot_k1.log`](logs/dit_xl_zeroshot_k1.log) — DiT-XL-2-256 zero-shot probe, K=1 only run (32k frames). Backs [`reports/DIT_REPORT.md` §3.1](../reports/DIT_REPORT.md).
- [`logs/dit_xl_zeroshot_k4.log`](logs/dit_xl_zeroshot_k4.log) — Same DiT model, K=1 + K=4 with embedding cache. Backs [`reports/DIT_REPORT.md` §3.2](../reports/DIT_REPORT.md).
- [`logs/subset_pusht.log`](logs/subset_pusht.log) — Subset of `pusht_expert_train.h5` from 18 685 → 500 episodes (seed=42).

### Scripts

- [`scripts/subset_pusht.py`](scripts/subset_pusht.py) — Builds a 500-episode validation subset of LeRobot PushT, preserving Blosc(lz4) compression and remapping `episode_idx`.
