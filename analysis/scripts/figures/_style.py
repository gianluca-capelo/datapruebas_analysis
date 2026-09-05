"""Style, typography and saving helpers shared by every figure script."""
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers the "science" style

from src.config import BASE_DIR

FIGURES_DIR = os.path.join(BASE_DIR, "analysis", "figures")

PRINT_DPI = 300
TRIAL_DPI = 600
ANIMATION_DPI = 150

SLIDE_FONT_SCALE = 1.4

SLIDE_RCPARAMS = {
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
}

PANEL_RCPARAMS = {
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}


def use_science_style():
    """Apply the scienceplots look shared by every figure in this package."""
    plt.style.use(["science", "no-latex"])


def use_slide_style():
    """Science style with the larger typography used for slide figures."""
    use_science_style()
    plt.rcParams.update(SLIDE_RCPARAMS)


@dataclass(frozen=True)
class FontSizes:
    """Point sizes for one figure, scalable in a single step for slides."""

    label: int = 18
    tick: int = 15
    legend: int = 15
    target: int = 15

    def scaled(self, factor: float = SLIDE_FONT_SCALE) -> "FontSizes":
        return FontSizes(
            label=round(self.label * factor),
            tick=round(self.tick * factor),
            legend=round(self.legend * factor),
            target=round(self.target * factor),
        )


TRIAL_FONTS = FontSizes()


def lang_suffix(lang: str) -> str:
    """Filename suffix keeping the Spanish and English variants apart."""
    return "_es" if lang == "es" else ""


def add_lang_argument(parser, default: str = "es"):
    parser.add_argument("--lang", choices=["en", "es"], default=default,
                        help=f"Idioma de los textos de la figura (default: {default})")


def save_fig(fig, name: str, formats=("png", "pdf"), dpi: int = PRINT_DPI,
             tight: bool = True) -> str:
    """Write `fig` to FIGURES_DIR as `name.<fmt>` for each requested format.

    ``tight=False`` preserves the axes rectangle exactly where it was placed,
    which the ``--aligned`` permutation figures depend on. Passing
    ``bbox_inches=None`` would not do it: matplotlib then falls back to
    scienceplots' ``savefig.bbox = tight``, hence the explicit full-canvas box.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    base = os.path.join(FIGURES_DIR, name)
    bbox = "tight" if tight else fig.bbox_inches
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", dpi=dpi, bbox_inches=bbox)
    extras = "".join(f"  (+ .{fmt})" for fmt in formats[1:])
    print(f"Saved -> {base}.{formats[0]}{extras}")
    return base
