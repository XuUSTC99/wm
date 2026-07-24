"""Shared plotting style for the AAAI figures.

Two things this fixes across every figure:

1. Typography. AAAI requires the paper to be set in Times Roman or Nimbus --
   Computer Modern is allowed for mathematics only -- and requires every font
   to be embedded, including inside figures, with no Type 3 anywhere. The body
   already satisfies this (NimbusRomNo9L); matplotlib's default DejaVu Sans
   did not match it, so figure text read as a different document pasted in.

   Figures therefore set in Nimbus Roman too, registered from the system OTF
   because matplotlib does not index it by default.

   GLYPH COVERAGE IS THE TRAP HERE. Nimbus Roman is a Times clone with a
   Times-era character set. It has no U+2713 CHECK MARK and no U+1E91 Z WITH
   CIRCUMFLEX. Neither matplotlib nor cairosvg warns: matplotlib draws a
   hollow .notdef box, cairosvg draws nothing at all. So:
     - the scan figure marks its seed-robust gain with U+2021 DOUBLE DAGGER,
       not a check mark;
     - the architecture SVG composes z-hat from a z plus a positioned
       circumflex rather than using the precomposed character.
   Check any new glyph against `fc-list :charset=XXXX` before using it.

2. Palette. Muted blue/sienna rather than saturated Okabe-Ito. The hues are
   still the Okabe-Ito axis (short-wavelength vs long-wavelength), which is
   what keeps them separable under deuteranopia and protanopia -- only the
   chroma comes down. Any change here must be re-checked with cvd_check.py;
   desaturation shrinks the margin that makes the pair safe.

Import and call `apply()` before creating figures.
"""
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# The URW base-35 Nimbus Roman that the AAAI template uses for body text.
# matplotlib does not index /usr/share/fonts/opentype by default, so register
# the family explicitly; without this the serif request silently falls back.
# System path first; on hosts without the urw-base35 package the same OTFs are
# picked up from the per-user font dir (install: drop NimbusRoman-*.otf there).
_NIMBUS_GLOBS = [
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-*.otf",
    os.path.expanduser("~/.local/share/fonts/urw-base35/NimbusRoman-*.otf"),
    os.path.expanduser("~/.fonts/NimbusRoman-*.otf"),
]
for _g in _NIMBUS_GLOBS:
    for _f in glob.glob(_g):
        try:
            fm.fontManager.addfont(_f)
        except Exception:
            pass
NIMBUS_AVAILABLE = any(f.name == "Nimbus Roman" for f in fm.fontManager.ttflist)

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
        "font.serif": ["Nimbus Roman", "DejaVu Serif"],
        # Math in the figures is limited to rho and the up/down arrows, which
        # Nimbus carries; "custom" keeps them in the text face instead of
        # pulling in a second family for two symbols.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Roman",
        "mathtext.it": "Nimbus Roman:italic",
        "mathtext.bf": "Nimbus Roman:bold",
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
