"""Appendix figure 6 -- the frozen-DINOv2 injection scan, 9 variants x 3 domains.

Same drawing routine, verdict rule, and row merge as paper figure 2
(scan_heatmap.py + the a=g/grounded pooling documented in dino_resolve.py), so
the two scans read identically: warm = worse, blue = better, hatched = the
cell's seeds overlap the baseline's.

Every value is parsed from raw_data/runs/dinowm at draw time; the previous
version of this figure classified cells by a +/-5% ratio threshold and marked
seed counts with daggers -- both replaced by the shared seed-separation rule.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dino_resolve import ROWS, DOMS, draws, baseline_draws
import scan_heatmap

HERE = pathlib.Path(__file__).resolve().parent
outs = [HERE / "fig17_dinowm_scan",
        HERE.parent / "paper" / "figures" / "fig17_dinowm_scan"]

scan_heatmap.draw(
    [(tuple(keys), lab) for keys, lab in ROWS],
    get_draws=lambda keys, j: draws(list(keys), DOMS[j]),
    get_base=lambda j: baseline_draws(DOMS[j]),
    outs=outs,
    cbar_label="nMSE / baseline  (both three-seed means)",
)
