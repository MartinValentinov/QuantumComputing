"""Assigning 2D coordinates to a molecular graph.

Rings first, then chains, then relaxation. Two details do most of the work,
and both were wrong in the first version of this file:

**Fused rings must be pinned, not approximated.** When a new ring shares an
edge with one already placed, its two shared atoms already have coordinates.
Generating a fresh polygon and hoping it lines up produces the mangled
bicyclics you get in bad depictions. Instead the new ring's centre is derived
from the shared edge, and the remaining atoms are stepped around that centre
starting from a known atom in the direction that lands the second known atom
exactly where it already is.

**Substituents must point away from everything already attached.** Taking the
direction from one arbitrary placed neighbour sends ring substituents straight
back across the ring. Summing the unit vectors to every placed neighbour and
going the opposite way is what makes a substituent radiate outward.
"""

from __future__ import annotations

import math

from qc_model.smiles import Mol, find_rings, parse

BOND_LENGTH = 1.0
TAU = 2 * math.pi


def _ring_systems(rings: list[list[int]]) -> list[list[list[int]]]:
    """Group rings sharing at least one bond into fused systems."""
    systems: list[list[list[int]]] = []
    for ring in rings:
        rs = set(ring)
        touching = [s for s in systems if any(len(rs & set(r)) >= 2 for r in s)]
        if not touching:
            systems.append([ring])
            continue
        combined = [ring]
        for s in touching:
            combined += s
            systems.remove(s)
        systems.append(combined)
    return systems


def _order_ring(mol: Mol, ring: list[int]) -> list[int]:
    """Walk the ring so consecutive entries are bonded."""
    ring_set = set(ring)
    ordered = [ring[0]]
    seen = {ring[0]}
    while len(ordered) < len(ring):
        nxt = next(
            (n for n in mol.atoms[ordered[-1]].neighbors if n in ring_set and n not in seen),
            None,
        )
        if nxt is None:
            break
        ordered.append(nxt)
        seen.add(nxt)
    return ordered


def _circumradius(n: int) -> float:
    return BOND_LENGTH / (2 * math.sin(math.pi / n)) if n > 2 else BOND_LENGTH


def _apothem(n: int) -> float:
    return BOND_LENGTH / (2 * math.tan(math.pi / n)) if n > 2 else 0.5


def _norm_angle(a: float) -> float:
    while a > math.pi:
        a -= TAU
    while a < -math.pi:
        a += TAU
    return a


def _layout_ring_system(
    mol: Mol, system: list[list[int]], pos: dict[int, tuple[float, float]],
    origin: tuple[float, float],
) -> None:
    remaining = [_order_ring(mol, r) for r in system]
    centroids: dict[int, tuple[float, float]] = {}

    remaining.sort(key=len, reverse=True)
    first = remaining.pop(0)
    n = len(first)
    r = _circumradius(n)
    cx, cy = origin
    for k, atom in enumerate(first):
        angle = math.pi / 2 + k * TAU / n
        pos[atom] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    centroids[id(first)] = (cx, cy)
    placed_rings = [first]

    progress = True
    while remaining and progress:
        progress = False
        for ring in list(remaining):
            known = [a for a in ring if a in pos]
            if len(known) < 2:
                continue

            edge = None
            for i in range(len(ring)):
                a, b = ring[i], ring[(i + 1) % len(ring)]
                if a in pos and b in pos:
                    edge = (a, b)
                    break
            if edge is None:
                continue
            a, b = edge

            ax, ay = pos[a]
            bx, by = pos[b]
            mx, my = (ax + bx) / 2, (ay + by) / 2

            parent = min(
                (pr for pr in placed_rings if a in pr and b in pr),
                key=lambda pr: len(pr),
                default=None,
            )
            ref = centroids.get(id(parent)) if parent else None
            if ref is None:
                others = [p for i2, p in pos.items() if i2 not in (a, b)]
                ref = (sum(p[0] for p in others) / len(others),
                       sum(p[1] for p in others) / len(others)) if others else (mx, my)

            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            if (mx - ref[0]) * nx + (my - ref[1]) * ny < 0:
                nx, ny = -nx, -ny

            nring = len(ring)
            ccx = mx + nx * _apothem(nring)
            ccy = my + ny * _apothem(nring)

            ang_a = math.atan2(ay - ccy, ax - ccx)
            ang_b = math.atan2(by - ccy, bx - ccx)
            step = TAU / nring
            direction = 1.0 if _norm_angle(ang_b - ang_a) > 0 else -1.0

            start = ring.index(a)
            walk = ring[start:] + ring[:start]
            if walk[1] != b:
                walk = [walk[0]] + walk[1:][::-1]

            radius = _circumradius(nring)
            for k, atom in enumerate(walk):
                if atom in pos:
                    continue
                angle = ang_a + direction * k * step
                pos[atom] = (ccx + radius * math.cos(angle),
                             ccy + radius * math.sin(angle))

            centroids[id(ring)] = (ccx, ccy)
            placed_rings.append(ring)
            remaining.remove(ring)
            progress = True

    for k, ring in enumerate(remaining):
        n = len(ring)
        r = _circumradius(n)
        cx2, cy2 = origin[0] + 3.2 * (k + 1), origin[1] - 2.6
        for j, atom in enumerate(ring):
            pos.setdefault(atom, (cx2 + r * math.cos(j * TAU / n),
                                  cy2 + r * math.sin(j * TAU / n)))


def _layout_chains(mol: Mol, pos: dict[int, tuple[float, float]]) -> None:
    """Grow acyclic atoms outward, away from whatever is already attached."""
    if not pos:
        pos[mol.atoms[0].index] = (0.0, 0.0)

    queue = list(pos)
    visited = set(pos)

    while queue:
        current = queue.pop(0)
        cx, cy = pos[current]

        pending = [n for n in mol.atoms[current].neighbors if n not in visited]
        if not pending:
            continue

        sx = sy = 0.0
        for n in mol.atoms[current].neighbors:
            if n in pos:
                px, py = pos[n]
                d = math.hypot(px - cx, py - cy) or 1.0
                sx += (px - cx) / d
                sy += (py - cy) / d
        base = math.atan2(-sy, -sx) if (abs(sx) + abs(sy)) > 1e-6 else 0.0

        spread = math.pi / 3
        count = len(pending)
        offsets = [(i - (count - 1) / 2) * spread for i in range(count)]

        for neighbor, off in zip(pending, offsets):
            angle = base + off
            pos[neighbor] = (cx + BOND_LENGTH * math.cos(angle),
                             cy + BOND_LENGTH * math.sin(angle))
            visited.add(neighbor)
            queue.append(neighbor)


def _relax(mol: Mol, pos: dict[int, tuple[float, float]], iterations: int = 260) -> None:
    """Springs on bonds, repulsion between everything else.

    Ring atoms barely move -- stage one already placed them correctly, and
    letting relaxation pull on them turns regular hexagons into blobs.
    """
    ring_atoms = {a.index for a in mol.atoms if a.in_ring}
    bonded = {(min(b.a, b.b), max(b.a, b.b)) for b in mol.bonds}
    indices = list(pos)

    for step in range(iterations):
        cooling = 1.0 - step / iterations
        force = {i: [0.0, 0.0] for i in indices}

        for a, b in bonded:
            ax, ay = pos[a]
            bx, by = pos[b]
            dx, dy = bx - ax, by - ay
            dist = math.hypot(dx, dy) or 1e-6
            pull = (dist - BOND_LENGTH) * 0.55
            ux, uy = dx / dist, dy / dist
            force[a][0] += pull * ux
            force[a][1] += pull * uy
            force[b][0] -= pull * ux
            force[b][1] -= pull * uy

        for idx, i in enumerate(indices):
            ix, iy = pos[i]
            for j in indices[idx + 1:]:
                if (min(i, j), max(i, j)) in bonded:
                    continue
                jx, jy = pos[j]
                dx, dy = jx - ix, jy - iy
                d2 = dx * dx + dy * dy
                if d2 > 2.9 or d2 < 1e-9:
                    continue
                dist = math.sqrt(d2)
                push = (1.7 - dist) * 0.30
                ux, uy = dx / dist, dy / dist
                force[i][0] -= push * ux
                force[i][1] -= push * uy
                force[j][0] += push * ux
                force[j][1] += push * uy

        for i in indices:
            stiffness = 0.03 if i in ring_atoms else 0.45
            fx, fy = force[i]
            x, y = pos[i]
            pos[i] = (x + fx * stiffness * cooling, y + fy * stiffness * cooling)


def _system_of(rings: list[list[int]]) -> dict[int, int]:
    """atom index -> ring-system id."""
    out: dict[int, int] = {}
    for sid, system in enumerate(_ring_systems(rings)):
        for ring in system:
            for i in ring:
                out[i] = sid
    return out


def layout(smiles: str) -> tuple[Mol, dict[int, tuple[float, float]]]:
    """SMILES -> (molecule, {atom index: (x, y)}).

    One breadth-first walk over the whole molecule, placing each ring system
    as a rigid unit the moment the walk first reaches it, oriented to grow
    away from whatever placed it.

    Laying every ring system out at a fixed offset and letting relaxation pull
    them together does not work. Ring atoms are damped hard on purpose -- they
    are the part a chemist reads first and must not deform -- so the springs
    between systems never win. Imatinib came out stretched into a flat line
    (aspect ratio 6.7, bond-length CV 0.53) instead of a compact drawing.
    """
    mol = parse(smiles)
    rings = [_order_ring(mol, r) for r in find_rings(mol)]
    systems = _ring_systems(rings)
    sys_of = _system_of(rings)

    local: dict[int, dict[int, tuple[float, float]]] = {}
    for sid, system in enumerate(systems):
        coords: dict[int, tuple[float, float]] = {}
        _layout_ring_system(mol, system, coords, origin=(0.0, 0.0))
        local[sid] = coords

    pos: dict[int, tuple[float, float]] = {}

    def place_system(sid: int, anchor: int, at: tuple[float, float], heading: float) -> None:
        coords = local[sid]
        if anchor not in coords:
            return
        ax, ay = coords[anchor]
        cx = sum(p[0] for p in coords.values()) / len(coords)
        cy = sum(p[1] for p in coords.values()) / len(coords)
        rot = heading - math.atan2(cy - ay, cx - ax)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        for idx, (x, y) in coords.items():
            dx, dy = x - ax, y - ay
            pos[idx] = (at[0] + dx * cos_r - dy * sin_r,
                        at[1] + dx * sin_r + dy * cos_r)

    if systems:
        biggest = max(range(len(systems)), key=lambda s: len(local[s]))
        place_system(biggest, next(iter(local[biggest])), (0.0, 0.0), 0.0)
    else:
        pos[mol.atoms[0].index] = (0.0, 0.0)

    for _ in range(mol.n_atoms + 2):
        progressed = False
        for atom in mol.atoms:
            if atom.index in pos:
                continue
            parents = [n for n in atom.neighbors if n in pos]
            if not parents:
                continue
            parent = parents[0]
            px, py = pos[parent]

            used = [
                math.atan2(pos[n][1] - py, pos[n][0] - px)
                for n in mol.atoms[parent].neighbors
                if n in pos
            ]
            base = (used[0] + math.pi) if used else 0.0
            best_ang, best_gap = base, -1.0
            for k in range(24):
                cand = base + (k - 12) * (math.pi / 12)
                gap = min((abs(_norm_angle(cand - u)) for u in used), default=math.pi)
                if gap > best_gap:
                    best_gap, best_ang = gap, cand

            target = (px + math.cos(best_ang), py + math.sin(best_ang))
            sid = sys_of.get(atom.index)
            if sid is not None and atom.index in local[sid]:
                place_system(sid, atom.index, target, best_ang)
            else:
                pos[atom.index] = target
            progressed = True
        if not progressed:
            break

    stray = 0
    for atom in mol.atoms:
        if atom.index not in pos:
            pos[atom.index] = (-4.0 - stray * 1.5, -4.0)
            stray += 1

    _relax(mol, pos)
    return mol, pos
