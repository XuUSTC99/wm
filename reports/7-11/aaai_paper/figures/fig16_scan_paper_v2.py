"""Paper figure 2 -- the LeWM injection scan, 9 variants x 3 domains.

Every number is read from the per-seed rollout logs by scan_resolve.py --
nothing here is transcribed. Both the cells and the baseline are three-seed
means (the merged a=g row: four draws, see below).

Why 9 rows and not 10: [dyn] strict a=g and [free] grounded were launched as
two arms but are the SAME configuration -- a fixed slot with a learned-constant
kinematic head under ground-truth labels (their config.yaml files differ only
in the run name). The row is shown once, labelled with both roles; it pools
the draws of both launches (two independent seed-3072 runs + the 1234/42
top-ups). The label-free prior keeps its labeled control -- that control IS
this row.

Style + verdict rule live in scan_heatmap.py, shared with the DINOv2 scan
(fig 6) so the two figures cannot drift apart.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scan_resolve import CELLS, BASELINE, resolve
import scan_heatmap

DOMS = ["uniform", "parabola", "collision"]

ROWS = [
    ("[slot] structpos", "[slot] structpos"),
    ("[slot] +reweight (w=30)", "[slot] +reweight (w=30)"),
    ("[slot] +velocity", "[slot] +velocity"),
    ("[probe] probe", "[probe] probe"),
    ("[probe] +slot", "[probe] +slot"),
    ("[dyn] free MLP", "[dyn] free MLP"),
    # merged row: same configuration under two names; CELLS already pools the
    # duplicate seed-3072 runs for this arm (4 draws per domain)
    ("[dyn] strict a=g", "[dyn] strict a=g\n(= [free] grounded)"),
    ("[cons] consistency", "[cons] consistency"),
    ("[free] label-free", "[free] label-free"),
]

HERE = pathlib.Path(__file__).resolve().parent
outs = [HERE / "fig16_scan_paper_v2",
        HERE.parent / "paper" / "figures" / "fig16_scan_paper"]

scan_heatmap.draw(
    ROWS,
    get_draws=lambda key, j: resolve(CELLS[(key, DOMS[j])], DOMS[j]),
    get_base=lambda j: resolve(BASELINE[DOMS[j]], DOMS[j]),
    outs=outs,
    cbar_label="nMSE / baseline  (both three-seed means)",
)
