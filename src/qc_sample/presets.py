"""Ready-made panels to run the pipeline against.

Six samples covering the diseases in the graph plus a healthy control, each
generated from that disease's real target associations and then frozen with a
fixed seed so the same sample gives the same answer every time.

The control matters more than it looks. A pipeline that always returns a
confident disease is useless -- feeding it a panel with nothing wrong is the
cheapest way to see whether its confidence means anything.
"""

from __future__ import annotations

import sqlite3

PRESETS = [
    {
        "id": "pdac",
        "label": "Pancreatic sample",
        "note": "Clear tumour signal",
        "efo_id": "MONDO_0005184",
        "strength": 1.0,
        "seed": 7,
    },
    {
        "id": "parkinson",
        "label": "Parkinson sample",
        "note": "Clear neurological signal",
        "efo_id": "MONDO_0005180",
        "strength": 1.0,
        "seed": 5,
    },
    {
        "id": "diabetes",
        "label": "Diabetes sample",
        "note": "Clear metabolic signal",
        "efo_id": "MONDO_0005015",
        "strength": 1.0,
        "seed": 11,
    },
    {
        "id": "parkinson_early",
        "label": "Early Parkinson",
        "note": "Weak signal, harder call",
        "efo_id": "MONDO_0005180",
        "strength": 0.35,
        "seed": 21,
    },
    {
        "id": "neuropathy",
        "label": "Neuropathy sample",
        "note": "Overlaps other signatures",
        "efo_id": "MONDO_0011468",
        "strength": 1.0,
        "seed": 3,
    },
    {
        "id": "healthy",
        "label": "Healthy control",
        "note": "Nothing abnormal — the answer should be low confidence",
        "efo_id": None,
        "strength": 0.0,
        "seed": 42,
    },
]


def build(conn: sqlite3.Connection, preset_id: str) -> dict:
    """Materialise one preset into a panel."""
    from qc_sample.panel import disease_signatures, simulate_panel

    spec = next((p for p in PRESETS if p["id"] == preset_id), None)
    if spec is None:
        raise ValueError(f"unknown preset {preset_id!r}")

    if spec["efo_id"] is None:
        import random

        rng = random.Random(spec["seed"])
        proteins = sorted(
            {s for sig in disease_signatures(conn).values() for s in sig["proteins"]}
        )
        from qc_sample.panel import HOUSEKEEPING

        panel = {s: round(rng.gauss(0, 0.35), 3) for s in proteins + HOUSEKEEPING}
        return {
            "id": spec["id"], "label": spec["label"], "note": spec["note"],
            "panel": panel, "n_proteins": len(panel),
            "truth": {"name": "healthy control", "efo_id": None, "strength": 0.0},
        }

    out = simulate_panel(
        conn, spec["efo_id"], strength=spec["strength"], seed=spec["seed"]
    )
    return {
        "id": spec["id"], "label": spec["label"], "note": spec["note"],
        "panel": out["panel"], "n_proteins": out["n_proteins"], "truth": out["truth"],
    }


def build_all(conn: sqlite3.Connection) -> list[dict]:
    """Every preset, skipping any whose disease is not in the graph."""
    out = []
    for spec in PRESETS:
        try:
            out.append(build(conn, spec["id"]))
        except ValueError:
            continue
    return out


def highlights(panel: dict[str, float], n: int = 8) -> list[dict]:
    """The most abnormal proteins, for a thumbnail of the sample."""
    rows = sorted(panel.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [
        {"symbol": s, "log2": v, "fold": round(2 ** v, 2),
         "direction": "up" if v > 0 else "down"}
        for s, v in rows
    ]
