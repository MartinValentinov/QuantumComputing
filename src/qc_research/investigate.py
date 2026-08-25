"""The investigation loop: disease in, ranked and justified shortlist out.

Structure, and why it is two passes rather than one:

    pass 1  SURVEY     cheap. Score every target in the graph from data we
                       already have -- association, tractability, chemistry.
    pass 2  DEEP DIVE  expensive. Build a full dossier for the top N only.
                       This is where safety data (essentiality, LOEUF,
                       liabilities) first becomes available.
    pass 3  RE-RANK    the deep dive routinely disqualifies things the survey
                       liked. Re-score with the new evidence and report the
                       movement, because *what changed* is the finding.

There is no LLM here. Ranking is an explicit weighted rubric you can read,
tune, and disagree with; every component traces to a named database field.
The output is deliberately structured so a language model can write narrative
over it later -- but the judgement stays in code, where it can be tested.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field

from qc_research import db, dossier, queries
from qc_research.pipeline import IngestConfig, ingest_disease
from qc_research.sources import opentargets

# ---------------------------------------------------------------------------
# scoring rubric
# ---------------------------------------------------------------------------

# Each component is normalised to 0..1. Strategies differ only in how they
# weight them -- which is the honest way to express portfolio strategy.
STRATEGIES: dict[str, dict[str, float]] = {
    # Find opportunities nobody has taken. Novelty and evidence dominate.
    "novel":    {"evidence": 0.30, "tractability": 0.20, "chemistry": 0.10,
                 "novelty": 0.30, "safety": 0.10},
    # Shortest path to a molecule. Existing chemistry and precedent dominate.
    "fast":     {"evidence": 0.20, "tractability": 0.15, "chemistry": 0.35,
                 "precedent": 0.25, "safety": 0.05},
    # No strong prior.
    "balanced": {"evidence": 0.30, "tractability": 0.20, "chemistry": 0.20,
                 "novelty": 0.10, "safety": 0.20},
}

TRACTABILITY_WEIGHT = {"SM": 1.0, "PR": 0.7, "AB": 0.6, "OC": 0.4}


def _evidence(score: float | None) -> float:
    """Association scores rarely exceed 0.6 in practice, so scale to that."""
    return min((score or 0.0) / 0.6, 1.0)


def _tractability(tract_json: str | None) -> float:
    mods = {t["modality"] for t in json.loads(tract_json or "[]")}
    return max((TRACTABILITY_WEIGHT.get(m, 0.0) for m in mods), default=0.0)


def _chemistry(n_potent: int) -> float:
    """Log scale: going 0 -> 10 compounds matters far more than 200 -> 400."""
    return min(math.log10(n_potent + 1) / 2.0, 1.0)


def _precedent(max_phase: float | None) -> float:
    return min((max_phase or 0.0) / 4.0, 1.0)


def _safety(profile: dict | None) -> tuple[float, list[str]]:
    """Start at 1.0 and subtract for each named liability."""
    if not profile:
        return 0.7, ["not yet profiled"]          # neutral-ish prior
    penalty, reasons = 0.0, []
    if profile.get("is_essential"):
        penalty += 0.35
        reasons.append("pan-essential")
    if profile.get("loeuf_bin") == 0:
        penalty += 0.20
        reasons.append("LOEUF bin 0")
    n_liab = len(profile.get("safety_liabilities") or [])
    if n_liab:
        penalty += min(0.10 * n_liab, 0.25)
        reasons.append(f"{n_liab} safety liabilities")
    n_dis = profile.get("n_associated_diseases") or 0
    if n_dis > 1000:
        penalty += 0.15
        reasons.append(f"pleiotropic ({n_dis} diseases)")
    return max(0.0, 1.0 - penalty), reasons


def score_target(row: dict, strategy: str, profile: dict | None = None) -> dict:
    """Return the composite score plus every component that produced it."""
    weights = STRATEGIES[strategy]

    safety, safety_reasons = _safety(profile)
    components = {
        "evidence": _evidence(row.get("association_score")),
        "tractability": _tractability(row.get("_tractability_json")),
        "chemistry": _chemistry(row.get("n_potent") or 0),
        "precedent": _precedent(row.get("highest_phase")),
        "safety": safety,
    }
    components["novelty"] = 1.0 - components["precedent"]

    total = sum(weights.get(k, 0.0) * v for k, v in components.items())
    return {
        "score": total,
        "components": components,
        "weights": weights,
        "safety_reasons": safety_reasons,
    }


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

@dataclass
class Investigation:
    disease: str = ""
    efo_id: str = ""
    strategy: str = "balanced"
    n_surveyed: int = 0
    survey: list[dict] = field(default_factory=list)
    shortlist: list[dict] = field(default_factory=list)
    dossiers: dict = field(default_factory=dict)
    movement: list[dict] = field(default_factory=list)
    indication_context: dict = field(default_factory=dict)
    next_actions: list[str] = field(default_factory=list)


def _survey(conn: sqlite3.Connection, efo_id: str) -> list[dict]:
    """Pass 1: score everything from data already in the graph."""
    sql = """
    SELECT
        t.symbol, t.name, t.ensembl_id, t.chembl_id, t.tractability,
        dt.score                                       AS association_score,
        COUNT(DISTINCT a.chembl_id)                    AS n_actives,
        SUM(CASE WHEN a.pchembl_value >= 7 THEN 1 ELSE 0 END) AS n_potent_raw,
        MAX(a.pchembl_value)                           AS best_pchembl,
        COUNT(DISTINCT m.chembl_id)                    AS n_clinical_drugs,
        MAX(m.max_phase)                               AS highest_phase
    FROM disease_target dt
    JOIN target t   ON t.ensembl_id = dt.ensembl_id
    LEFT JOIN activity  a ON a.target_chembl_id = t.chembl_id
    LEFT JOIN mechanism m ON m.target_chembl_id = t.chembl_id
    WHERE dt.efo_id = ?
    GROUP BY t.ensembl_id
    ORDER BY dt.score DESC
    """
    rows = []
    for r in conn.execute(sql, (efo_id,)):
        d = dict(r)
        d["_tractability_json"] = d.pop("tractability")
        # n_potent counts distinct compounds, not measurements.
        d["n_potent"] = conn.execute(
            "SELECT COUNT(DISTINCT chembl_id) AS n FROM activity "
            "WHERE target_chembl_id = ? AND pchembl_value >= 7",
            (d["chembl_id"],),
        ).fetchone()["n"] if d["chembl_id"] else 0
        rows.append(d)
    return rows


def investigate(
    conn: sqlite3.Connection,
    disease: str,
    strategy: str = "balanced",
    top_n: int = 5,
    auto_ingest: bool = True,
    skip_literature: bool = False,
    verbose: bool = True,
) -> Investigation:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}")

    inv = Investigation(strategy=strategy)

    def say(m: str) -> None:
        if verbose:
            print(m, flush=True)

    # ---- resolve, ingesting first if the graph has nothing ----
    row = conn.execute(
        "SELECT efo_id, name FROM disease WHERE name LIKE ?", (f"%{disease}%",)
    ).fetchone()
    if row is None:
        if not auto_ingest:
            raise ValueError(f"{disease!r} not in the graph; run `ingest` first")
        say(f"[setup] {disease!r} not in graph -- ingesting first")
        rep = ingest_disease(conn, disease, IngestConfig(), verbose=verbose)
        inv.disease, inv.efo_id = rep.disease, rep.efo_id
    else:
        inv.disease, inv.efo_id = row["name"], row["efo_id"]

    say(f"\n[pass 1] survey: scoring all targets for {inv.disease} ({strategy})")
    survey = _survey(conn, inv.efo_id)
    inv.n_surveyed = len(survey)
    if not survey:
        raise ValueError(f"no targets in the graph for {inv.disease}; run `ingest`")

    for row_ in survey:
        row_["survey_score"] = score_target(row_, strategy)
    survey.sort(key=lambda r: -r["survey_score"]["score"])
    inv.survey = survey
    say(f"          {len(survey)} targets scored")
    for r in survey[:top_n]:
        say(f"          {r['survey_score']['score']:.3f}  {r['symbol']}")

    # ---- pass 2: deep dive on the shortlist ----
    say(f"\n[pass 2] deep dive on top {top_n}")
    for r in survey[:top_n]:
        say(f"          profiling {r['symbol']}...")
        d = dossier.build(
            conn,
            r["symbol"],
            disease=inv.disease,
            skip_literature=skip_literature,
        )
        inv.dossiers[r["symbol"]] = d

    # ---- pass 3: re-rank with the safety data the dossier just surfaced ----
    say("\n[pass 3] re-ranking with safety data")
    rescored = []
    for r in survey[:top_n]:
        d = inv.dossiers[r["symbol"]]
        new = score_target(r, strategy, profile=d.profile)
        rescored.append({**r, "final_score": new, "dossier": d})

    rescored.sort(key=lambda r: -r["final_score"]["score"])

    old_order = [r["symbol"] for r in survey[:top_n]]
    new_order = [r["symbol"] for r in rescored]
    for pos, sym in enumerate(new_order):
        was = old_order.index(sym)
        if was != pos:
            inv.movement.append(
                {
                    "symbol": sym,
                    "from_rank": was + 1,
                    "to_rank": pos + 1,
                    "reason": ", ".join(
                        next(r for r in rescored if r["symbol"] == sym)["final_score"][
                            "safety_reasons"
                        ]
                    )
                    or "safety data now available",
                }
            )
    if inv.movement:
        for m in inv.movement:
            direction = "down" if m["to_rank"] > m["from_rank"] else "up"
            say(
                f"          {m['symbol']} moved {direction} "
                f"{m['from_rank']} -> {m['to_rank']} ({m['reason']})"
            )
    else:
        say("          no rank changes")

    inv.shortlist = rescored
    inv.indication_context = queries.failure_patterns(conn, inv.efo_id)
    inv.next_actions = _next_actions(inv)
    return inv


def _next_actions(inv: Investigation) -> list[str]:
    """Concrete follow-ups, derived from what the shortlist actually looks like."""
    actions = []
    for r in inv.shortlist[:3]:
        d = r["dossier"]
        sym = r["symbol"]
        chem = d.chemistry
        n_potent = (chem.get("potency") or {}).get("n_potent") or 0
        probes = [x for x in d.profile.get("chemical_probes", []) if x.get("high_quality")]

        if chem.get("n_compounds", 0) == 0:
            actions.append(
                f"{sym}: no chemical matter exists. Before anything else, check "
                f"whether a structure is available (PDB) -- without one, this is a "
                f"screening campaign, not a design problem."
            )
        elif n_potent < 10:
            actions.append(
                f"{sym}: only {n_potent} potent compounds. Pull the full ChEMBL "
                f"activity set at min_pchembl=5 and look for a tractable series."
            )
        elif d.clinical.get("n_approved"):
            actions.append(
                f"{sym}: {d.clinical['n_approved']} approved drug(s) already. The "
                f"question is differentiation -- read the resistance literature "
                f"before committing."
            )
        else:
            actions.append(
                f"{sym}: {n_potent} potent compounds and no approved drug. Best "
                f"candidate for a design loop -- start here."
            )

        if not probes and chem.get("n_compounds", 0) > 0:
            actions.append(
                f"{sym}: no high-quality chemical probe. Target validation will "
                f"rest on genetics alone."
            )

    ctx = inv.indication_context
    buckets = ctx.get("stop_reason_buckets") or {}
    if buckets:
        top = max(buckets, key=buckets.get)
        actions.append(
            f"Indication-level: the most common reason trials stop in "
            f"{inv.disease} is '{top}' ({buckets[top]} of "
            f"{ctx.get('n_stopped_with_reason', 0)} stated reasons). Design around "
            f"that before optimising potency."
        )
    return actions
