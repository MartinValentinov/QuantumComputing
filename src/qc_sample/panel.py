"""Protein panels: simulate one, read one, and work out what it points at.

A panel maps a protein symbol to its abundance relative to healthy tissue,
expressed as log2 fold change. Zero is normal, +1 is twice normal, -1 is half.
Log space is used throughout because protein abundance is multiplicative:
the distance from 1x to 2x is the same kind of step as 4x to 8x, and averaging
raw ratios lets one big number swamp everything else.

Matching a panel to a disease is a cosine similarity against signatures built
from Open Targets association scores. That is deliberately a simple method.
The interesting part is not the classifier, it is that every signature traces
back to evidence already in the graph, so a match can always be explained.
"""

from __future__ import annotations

import math
import random
import re
import sqlite3

ELEVATED = 1.4
SUPPRESSED = -1.1


def disease_signatures(conn: sqlite3.Connection) -> dict[str, dict]:
    """One expected panel per disease, from its target associations.

    A target with a strong association is expected to read high; the score
    scales how high. This is a caricature of real biology -- plenty of disease
    proteins go down, not up -- but it is a caricature drawn from real data.
    """
    out: dict[str, dict] = {}
    for row in conn.execute(
        """SELECT d.efo_id, d.name, t.symbol, dt.score
           FROM disease_target dt
           JOIN disease d ON d.efo_id = dt.efo_id
           JOIN target  t ON t.ensembl_id = dt.ensembl_id
           WHERE t.symbol IS NOT NULL
           ORDER BY dt.score DESC"""
    ):
        entry = out.setdefault(
            row["efo_id"], {"name": row["name"], "proteins": {}}
        )
        entry["proteins"][row["symbol"]] = float(row["score"] or 0.0)
    return out


HOUSEKEEPING = [
    "ACTB", "GAPDH", "TUBB", "RPLP0", "B2M", "PPIA", "YWHAZ", "SDHA",
    "ALB", "TF", "APOA1", "HP", "C3", "FGA", "IGHG1", "SERPINA1",
    "CRP", "LDHA", "ENO1", "PKM", "HSPA8", "EEF1A1", "CFL1", "VIM",
]


def simulate_panel(
    conn: sqlite3.Connection,
    efo_id: str,
    n_background: int = 22,
    noise: float = 0.35,
    strength: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Generate a plausible panel for a disease.

    `strength` scales how loudly the disease speaks: 1.0 is a clear case, 0.3
    is early or well-controlled. `noise` is assay variation, applied to every
    protein including the healthy ones, because a panel with clean zeros for
    the background is not something any real instrument produces.
    """
    rng = random.Random(seed)
    signatures = disease_signatures(conn)
    if efo_id not in signatures:
        raise ValueError(f"no signature for {efo_id!r} -- ingest the disease first")

    sig = signatures[efo_id]
    values: dict[str, float] = {}

    for symbol, score in sig["proteins"].items():
        direction = ELEVATED if rng.random() > 0.25 else SUPPRESSED
        values[symbol] = round(
            direction * (score / 0.6) * strength + rng.gauss(0, noise), 3
        )

    for symbol in HOUSEKEEPING[:n_background]:
        values.setdefault(symbol, round(rng.gauss(0, noise), 3))

    for other_id, other in signatures.items():
        if other_id == efo_id:
            continue
        for symbol in other["proteins"]:
            values.setdefault(symbol, round(rng.gauss(0, noise), 3))

    return {
        "panel": values,
        "truth": {"efo_id": efo_id, "name": sig["name"], "strength": strength},
        "n_proteins": len(values),
    }


_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_\-]{1,15})\s*[,:\t ]\s*(-?\d+\.?\d*)\s*$")


def parse_panel(text: str) -> dict[str, float]:
    """Read a panel from free text: 'KRAS 2.4', 'KRAS,2.4' or 'KRAS: 2.4'."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            values[m.group(1).upper()] = float(m.group(2))
    return values


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


MIN_ENRICHMENT = 1.6


def match_diseases(
    panel: dict[str, float], signatures: dict[str, dict], top_n: int = 5
) -> list[dict]:
    """Rank diseases by how well the panel matches their signature.

    Two numbers, because either alone is misleading:

      * **similarity** -- cosine against the expected pattern. Says whether the
        panel points the right way, and ignores magnitude entirely.
      * **enrichment** -- how much larger the deviations are on the signature's
        proteins than on the rest of the panel. Says whether there is anything
        there to point.

    A healthy control makes the case for the second. Pure assay noise scored a
    cosine similarity of 0.304 against one signature -- as high as a genuine
    tumour sample -- because a small vector pointing roughly the right way
    looks identical to a large one under cosine. Enrichment separates them:
    noise is by definition no stronger on signature proteins than anywhere
    else, so it sits near 1.0.

    Below MIN_ENRICHMENT the match is reported with `clear = False`. Callers
    are expected to say "no clear signal" rather than name a disease anyway.
    """
    magnitudes = [abs(v) for v in panel.values()]
    background = st_median(magnitudes) if magnitudes else 0.0
    background = max(background, 1e-6)

    scored = []
    for efo_id, sig in signatures.items():
        expected = {
            s: ELEVATED * (score / 0.6) for s, score in sig["proteins"].items()
        }
        sim = _cosine(panel, expected)

        measured = [abs(panel[s]) for s in sig["proteins"] if s in panel]
        signal = (sum(measured) / len(measured)) if measured else 0.0
        enrichment = signal / background

        scored.append(
            {
                "efo_id": efo_id,
                "name": sig["name"],
                "similarity": round(sim, 4),
                "enrichment": round(enrichment, 3),
                "signal": round(signal, 3),
                "background": round(background, 3),
                "proteins_measured": len(measured),
                "proteins_in_signature": len(sig["proteins"]),
            }
        )

    for r in scored:
        r["evidence"] = round(max(r["similarity"], 0.0) * max(r["enrichment"] - 1.0, 0.0), 4)

    scored.sort(key=lambda r: -r["evidence"])
    top = scored[:top_n]

    temperature = 0.25
    exps = [math.exp(r["evidence"] / temperature) for r in top]
    total = sum(exps) or 1.0
    for r, e in zip(top, exps):
        r["confidence"] = round(100 * e / total, 1)
        r["clear"] = bool(r["enrichment"] >= MIN_ENRICHMENT and r["similarity"] > 0.15)

    return top


def st_median(values: list[float]) -> float:
    import statistics

    return float(statistics.median(values)) if values else 0.0


def driving_targets(
    panel: dict[str, float], signature: dict, top_n: int = 8
) -> list[dict]:
    """Which measured proteins are actually driving the match.

    A protein counts when it is both abnormal in this sample and known to
    matter for this disease. Either alone is noise: plenty of proteins swing
    wildly for reasons unrelated to the illness, and plenty of well-evidenced
    disease genes sit at normal levels in any given patient.
    """
    rows = []
    for symbol, score in signature["proteins"].items():
        if symbol not in panel:
            continue
        deviation = panel[symbol]
        rows.append(
            {
                "symbol": symbol,
                "log2_fold_change": deviation,
                "fold_change": round(2 ** deviation, 2),
                "association": round(score, 3),
                "contribution": round(abs(deviation) * score, 3),
                "direction": "elevated" if deviation > 0 else "suppressed",
            }
        )
    rows.sort(key=lambda r: -r["contribution"])
    return rows[:top_n]
