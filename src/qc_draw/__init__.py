"""2D depiction of molecules: SMILES in, SVG out.

Written out rather than imported, like the rest of qc_model. Laying out a
molecule is a small optimisation problem -- rings want to be regular polygons,
chains want to zigzag at 120 degrees, and nothing wants to overlap -- and
watching those three rules fight each other is most of what makes a drawing
look right or wrong.
"""

from qc_draw.features import detect, lipinski
from qc_draw.layout import layout
from qc_draw.render import draw, to_svg

__all__ = ["layout", "to_svg", "draw", "detect", "lipinski"]
