# Citation Audit Report

**Date**: 2026-07-19
**Bib file**: `references.bib`
**Entries**: 22 total / 20 cited / 2 uncited
**Method**: three-layer audit (existence → metadata → context). Codex MCP was unavailable in this
environment, so the cross-model reviewer step was replaced by direct web/arXiv/DBLP verification
(WebSearch + WebFetch) performed per entry. Every finding below carries a verifying URL.

## Summary

| Verdict | Count | Entries |
|---------|-------|---------|
| KEEP    | 15    | all remaining cited entries |
| FIX     | 5     | `maes2025lewm`, `hafner2023dreamer`, `kang2024how`, `peper2025principles`, `zahorodnii2025deepsup` |
| REPLACE | 0     | — |
| REMOVE  | 0     | — |

**Existence: 20/20 PASS.** No hallucinated citations, no phantom DOIs, no anonymous placeholders.
**Context: PASS.** Every citation is used for a claim the cited paper actually establishes.
**Metadata: 5 FIX** — all applied (see below); no wrong-context or fabricated entries.

## Applied fixes

### 1. `maes2025lewm` — invented title (most severe)
The bib gave the title as *"le-wm: A Latent Embedding World Model"*. The repository's actual title is
**"LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"** — LeWM
expands to **LeWorldModel**, not "Latent Embedding World Model". This is the paper's own backbone, so
a reviewer following the link would have hit the discrepancy immediately.
Verified: <https://github.com/lucas-maes/le-wm>. **Fixed** (title corrected; paper body never expands
the acronym, so no prose change was needed).

### 2. `hafner2023dreamer` — preprint cited; published in Nature with a different title
Cited as CoRR abs/2301.04104 (2023), *"Mastering Diverse Domains through World Models"*. The paper is
published as **Nature 640:647–653 (2025)** under the changed title *"Mastering diverse **control
tasks** through world models"* (doi 10.1038/s41586-025-08744-2) — a textbook version/title-drift case.
Verified: <https://www.nature.com/articles/s41586-025-08744-2>. **Fixed** (now the Nature record;
renders in-text as "Hafner et al. 2025").

### 3. `kang2024how` (PhyWorld) — preprint cited; published at ICML 2025
Our primary benchmark, cited 6×, was a bare `@misc` arXiv 2411.02385 (2024). Published at
**ICML 2025**. Verified: <https://arxiv.org/abs/2411.02385>, <https://phyworld.github.io/>.
**Fixed** (now `@inproceedings`, ICML 2025; renders as "Kang et al. 2025").

### 4. `peper2025principles` — preprint cited; published in PMLR v288
Published in **Proceedings of the International Conference on Neuro-symbolic Systems (NeuS) 2025**,
PMLR v288, pp. 66–89. Verified: <https://proceedings.mlr.press/v288/peper25a.html>.
**Fixed** (now the PMLR record).

### 5. `zahorodnii2025deepsup` — workshop venue not recorded
Real and correctly attributed (Andrii Zahorodnii, MIT; arXiv 2504.03861), but it is a contribution to
the **ICLR 2025 Workshop on World Models** — worth recording, since a workshop-vetted source carries
more weight than a bare preprint for the load-bearing "deep supervision works in a compact state
latent" claim. Verified: <https://arxiv.org/abs/2504.03861>. **Fixed** (`howpublished` added).

## Verified clean (no action)

| Entry | Check |
|---|---|
| `nie2026physjepa` | arXiv 2606.16076 exists (15 Jun 2026; Nie, Liu, Guo, Su). Confirmed to be **numerical multivariate time series** (Jena Climate, Traffic) — which validates the scope wording used in §1/§2. |
| `mao2024piwm` | arXiv 2412.12870, current v6 title matches the bib exactly; authors Mao, Umasudhan, Ruchkin; still a preprint, so `@misc` is correct. (An intermediate v3 carried a different title — checked and not an issue at v6.) |
| `balestriero2025lejepa` | arXiv 2511.08544, Balestriero & LeCun. SIGReg = Sketched Isotropic Gaussian Regularization; our "anti-collapse regularizer" description is accurate. |
| `bear2021physion`, `tung2023physionpp`, `assran2023ijepa`, `bardes2024vjepa`, `ha2018world`, `bengio2015scheduled`, `venkatraman2015improving`, `lamb2016professor`, `greydanus2019hamiltonian`, `oquab2024dinov2`, `zhou2025dinowm`, `lecun2022path` | DBLP/verified records; metadata consistent. |

Note on `lamb2016professor`: the key names Lamb but the first author is **Goyal** — this is correct
(Professor Forcing's author order is Goyal, Lamb, Zhang, Zhang, Courville, Bengio) and it renders
in-text as "Goyal et al. 2016". Key name is internal only; no action needed.

## Uncited entries — pruned

Two bib entries were never cited and have been **removed** (bib: 22 → 20 entries, matching the 20
cited keys exactly):

- `ball2021augmented` — Augmented World Models (ICML 2021).
- `jin2024piaug` — PIAug (arXiv 2311.00815). The key was doubly wrong: no author is named "Jin"
  (authors are Maheshwari, Wang, Triest, Sivaprakasam, Aich, Rogers, Gregory, Scherer) and the key
  said 2024 while the entry's own `year` was 2023.

Post-prune rebuild: `bibtex` clean, `main.bbl` contains exactly 20 `\bibitem`s, 0 undefined
citations, 12 pages unchanged.

## Previously fixed in this session (context layer)

One genuine **wrong-context** citation was caught and corrected before this audit run: §1 grouped
`nie2026physjepa` with `zahorodnii2025deepsup` under "the supervised quantities occupy a large
fraction of the representation". The ~38% figure is documented only for Zahorodnii (3 of 8 latent
dims); no such figure exists for Phys-JEPA, whose distinguishing property is its **data modality**
(numerical time series, not visual latents). The two are now cited separately for what each actually
establishes.

## Verification

- `pdflatex` + `bibtex` + 2× `pdflatex`: zero errors, **0 undefined citations**, 12 pages unchanged.
- In-text rendering spot-checked: "Kang et al. 2025", "Hafner et al. 2025", "Maes 2025 — LeWorldModel…".
- Backup of the pre-audit bib: `references.bib.bak`.
