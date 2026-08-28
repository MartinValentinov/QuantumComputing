"""Coordinates -> SVG.

Conventions a chemist expects, and which the drawing is wrong without:

  * Carbons are not labelled. A vertex *is* a carbon. Labelling them turns a
    readable skeleton into alphabet soup.
  * Heteroatoms are labelled and coloured by the CPK convention -- N blue,
    O red, S yellow, halogens green.
  * Double bonds are two parallel lines, the second inset toward the ring
    centre when the bond is in a ring, which is how they are drawn by hand.
  * Aromatic rings get an inner arc rather than alternating double bonds,
    because the parser has already decided they are delocalised.
  * Implicit hydrogens ride along with their heteroatom label (OH, NH2).
"""

from __future__ import annotations

import math

from qc_model.smiles import Mol

ELEMENT_COLOR = {
    "N": "#2B55C4", "O": "#C0341F", "S": "#B8940B", "P": "#D07A18",
    "F": "#2E9E5B", "Cl": "#2E9E5B", "Br": "#8C3A1E", "I": "#7A3FA8",
    "B": "#C98A6A", "Si": "#7A7A7A", "Se": "#8C7A1E", "Na": "#6A4FBF",
    "K": "#6A4FBF", "Zn": "#5C7A8C", "Fe": "#B5651D",
}

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _label_for(mol: Mol, i: int) -> str | None:
    """What to write at this vertex, or None to leave it bare."""
    atom = mol.atoms[i]
    if atom.element == "C" and atom.charge == 0 and len(atom.neighbors) > 0:
        return None
    text = atom.element
    if atom.n_hydrogens == 1:
        text += "H"
    elif atom.n_hydrogens > 1:
        text += "H" + str(atom.n_hydrogens).translate(SUB)
    if atom.charge:
        text += "+" if atom.charge > 0 else "−"
        if abs(atom.charge) > 1:
            text += str(abs(atom.charge))
    return text


def to_svg(
    mol: Mol,
    pos: dict[int, tuple[float, float]],
    width: int = 420,
    height: int = 320,
    scale: float = 34.0,
    highlight: set[int] | None = None,
    animate: bool = False,
    background: str | None = None,
) -> str:
    """Render to a self-contained SVG string.

    `animate=True` draws the skeleton on stroke by stroke, which reads as the
    molecule being built rather than appearing at once.
    """
    highlight = highlight or set()
    if not pos:
        return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"></svg>'

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)

    margin = 26
    fit = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y, scale)
    cx_data, cy_data = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    def project(i: int) -> tuple[float, float]:
        x, y = pos[i]
        return (width / 2 + (x - cx_data) * fit, height / 2 - (y - cy_data) * fit)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="chemical structure">'
    ]
    if background:
        parts.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')

    labels = {i: _label_for(mol, i) for i in pos}
    ring_centroid = _ring_centroids(mol, pos)

    bond_svg: list[str] = []
    for bond in mol.bonds:
        if bond.a not in pos or bond.b not in pos:
            continue
        x1, y1 = project(bond.a)
        x2, y2 = project(bond.b)

        x1, y1, x2, y2 = _trim(x1, y1, x2, y2,
                               9.0 if labels.get(bond.a) else 0.0,
                               9.0 if labels.get(bond.b) else 0.0)

        hot = bond.a in highlight and bond.b in highlight
        stroke = "var(--mol-hi, #C0341F)" if hot else "currentColor"
        w = 2.4 if hot else 1.7

        if bond.aromatic:
            bond_svg.append(_line(x1, y1, x2, y2, stroke, w))
            inner = _offset_toward(x1, y1, x2, y2, ring_centroid.get((bond.a, bond.b)), 4.2)
            if inner:
                bond_svg.append(_line(*inner, stroke, w * 0.75, dash="1 0"))
        elif bond.order == 2.0:
            centre = ring_centroid.get((bond.a, bond.b))
            if centre:
                inner = _offset_toward(x1, y1, x2, y2, centre, 4.6)
                bond_svg.append(_line(x1, y1, x2, y2, stroke, w))
                if inner:
                    bond_svg.append(_line(*inner, stroke, w))
            else:
                for sign in (-1, 1):
                    bond_svg.append(_line(*_parallel(x1, y1, x2, y2, sign * 2.6), stroke, w))
        elif bond.order == 3.0:
            bond_svg.append(_line(x1, y1, x2, y2, stroke, w))
            for sign in (-1, 1):
                bond_svg.append(_line(*_parallel(x1, y1, x2, y2, sign * 3.4), stroke, w))
        else:
            bond_svg.append(_line(x1, y1, x2, y2, stroke, w))

    group_open = '<g fill="none" stroke-linecap="round">'
    if animate:
        parts.append(group_open)
        for i, svg in enumerate(bond_svg):
            delay = i * 0.045
            parts.append(
                svg.replace(
                    "/>",
                    f'><animate attributeName="stroke-dashoffset" from="60" to="0" '
                    f'dur="0.42s" begin="{delay:.2f}s" fill="freeze"/></line>'
                ).replace("<line ", '<line stroke-dasharray="60" stroke-dashoffset="60" ')
            )
        parts.append("</g>")
    else:
        parts.append(group_open + "".join(bond_svg) + "</g>")

    label_delay = len(bond_svg) * 0.045
    parts.append(
        '<g font-family="ui-sans-serif, system-ui, sans-serif" font-size="13" '
        'font-weight="600" text-anchor="middle" dominant-baseline="central">'
    )
    for i, text in labels.items():
        if not text:
            continue
        x, y = project(i)
        color = ELEMENT_COLOR.get(mol.atoms[i].element, "currentColor")
        if i in highlight:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" '
                         f'fill="var(--mol-hi, #C0341F)" opacity="0.16"/>')
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" fill="var(--mol-bg, #fff)"/>'
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}"'
            + (f' opacity="0"><animate attributeName="opacity" from="0" to="1" '
               f'dur="0.3s" begin="{label_delay:.2f}s" fill="freeze"/>' if animate else ">")
            + f"{_esc(text)}</text>"
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _ring_centroids(mol: Mol, pos: dict) -> dict[tuple[int, int], tuple[float, float]]:
    """For each ring bond, the centre of a ring containing it -- so the second
    line of a double bond can be inset inward."""
    from qc_model.smiles import find_rings

    out: dict[tuple[int, int], tuple[float, float]] = {}
    for ring in find_rings(mol):
        placed = [i for i in ring if i in pos]
        if len(placed) < 3:
            continue
        cx = sum(pos[i][0] for i in placed) / len(placed)
        cy = sum(pos[i][1] for i in placed) / len(placed)
        ring_set = set(ring)
        for bond in mol.bonds:
            if bond.a in ring_set and bond.b in ring_set:
                out.setdefault((bond.a, bond.b), (cx, cy))
                out.setdefault((bond.b, bond.a), (cx, cy))
    return out


def _line(x1, y1, x2, y2, stroke, w, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>')


def _parallel(x1, y1, x2, y2, offset):
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length * offset, dx / length * offset
    return x1 + nx, y1 + ny, x2 + nx, y2 + ny


def _offset_toward(x1, y1, x2, y2, centre, amount):
    """Shift a bond line toward a ring centre, and shorten it slightly so the
    inner line reads as inside the ring."""
    if centre is None:
        return None
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if (centre[0] - mx) * nx - (centre[1] - my) * ny < 0:
        nx, ny = -nx, -ny

    trim = length * 0.16
    return (x1 + nx * amount + ux * trim, y1 + ny * amount + uy * trim,
            x2 + nx * amount - ux * trim, y2 + ny * amount - uy * trim)


def _trim(x1, y1, x2, y2, start, end):
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    return x1 + ux * start, y1 + uy * start, x2 - ux * end, y2 - uy * end


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def draw(smiles: str, **kwargs) -> str:
    """Convenience: SMILES straight to SVG."""
    from qc_draw.layout import layout

    mol, pos = layout(smiles)
    return to_svg(mol, pos, **kwargs)
