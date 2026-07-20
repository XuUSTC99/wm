"""Shared plotting style for the AAAI figures.

Two things this fixes across every figure:

1. Typography. The paper body sets in Nimbus Roman (a Times clone, `utmr8a`
   in the log); matplotlib's default is DejaVu Sans, so figure text used to
   read as a different document pasted in.

   DejaVu Serif is the choice here, not a closer Times clone, because the
   architecture diagram (fig1_architecture.svg, rendered by cairosvg) needs
   the SAME font and needs U+1E91 "z with circumflex". Nimbus Roman, Georgia,
   Latin Modern and STIXGeneral all lack that glyph, and cairosvg drops
   missing glyphs SILENTLY rather than falling back per character -- the hat
   simply vanishes from the predicted-latent box. DejaVu Serif is the only
   serif on this machine that carries it, so it wins on being one font across
   every figure. Verify glyph coverage before changing this.

2. Palette. Muted blue/sienna rather than saturated Okabe-Ito. The hues are
   still the Okabe-Ito axis (short-wavelength vs long-wavelength), which is
   what keeps them separable under deuteranopia and protanopia -- only the
   chroma comes down. Any change here must be re-checked with cvd_check.py;
   desaturation shrinks the margin that makes the pair safe.

Import and call `apply()` before creating figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- palette -----------------------------------------------------------
BLUE = "#3B6E92"   # cool, low chroma -- "better than baseline" / free rollout
SIENNA = "#B5714A"   # warm, low chroma -- "worse than baseline" / teacher forcing
INK = "#2B2B2B"   # body text
MUTED = "#6E6E6E"   # axis furniture, secondary labels
PAPER = "#F2EFE9"   # warm off-white: diverging midpoint, shaded bands
RULE = "#9A9A9A"   # spines, separators

# Diverging scale for the scan heatmap: blue (gain) -> paper (parity) -> sienna.
SCAN_CMAP = LinearSegmentedColormap.from_list("scan", [BLUE, PAPER, SIENNA])

# Categorical set for the four evaluation partitions (dose-response figure).
# These carry distinct markers as well, so they do not have to survive CVD on
# hue alone -- which is just as well, since ochre and grey converge under
# deuteranopia. Ordered light-to-dark so the judged partition reads as heaviest.
CAT = ["#9A9A94", "#C2934A", "#7BA3C4", "#3B6E92"]   # ID, r/m-OOD, v-OOD, both-OOD


def apply(base=8.0):
    """Set rcParams. `base` is the body font size in points at final scale.

    The figures are placed at \\linewidth in a two-column layout, so text set
    here appears at roughly its nominal size in the PDF; 8pt sits just under
    the 9pt caption, which is the convention these venues follow.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.8,
        "ytick.labelsize": base - 0.8,
        "legend.fontsize": base - 1.0,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.7,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,   # embed as TrueType, not Type 3 -- some venues reject Type 3
        "ps.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")):
    """Drop the spines that carry no information."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
