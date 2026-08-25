"""The payoff layer: questions the graph can answer that no single source can.

Each function here spans at least two sources. That is the test of whether
the integration was worth doing -- if a question can be answered by one API
call, it does not belong here.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter


def target_landscape(conn: sqlite3.Connection, efo_id: str, limit: int = 25) -> list[dict]:
    """Per target: association strength (Open Targets) x chemical matter (ChEMBL).

    The four columns together are a triage table. A high score with zero
    actives is an unexploited opportunity or an undruggable target -- the
    tractability field tells you which.
    """
    sql = """
    SELECT
        t.symbol,
        t.name,
        t.chembl_id,
        dt.score                                    AS association_score,
        COUNT(DISTINCT a.chembl_id)                 AS n_actives,
        MAX(a.pchembl_value)                        AS best_pchembl,
        COUNT(DISTINCT m.chembl_id)                 AS n_clinical_drugs,
        MAX(m.max_phase)                            AS highest_phase,
        t.tractability
    FROM disease_target dt
    JOIN target t   ON t.ensembl_id = dt.ensembl_id
    LEFT JOIN activity  a ON a.target_chembl_id = t.chembl_id
    LEFT JOIN mechanism m ON m.target_chembl_id = t.chembl_id
    WHERE dt.efo_id = ?
    GROUP BY t.ensembl_id
    ORDER BY dt.score DESC
    LIMIT ?
    """
    rows = []
    for r in conn.execute(sql, (efo_id, limit)):
        d = dict(r)
        tract = json.loads(d.pop("tractability") or "[]")
        d["small_molecule_tractable"] = any(t["modality"] == "SM" for t in tract)
        d["antibody_tractable"] = any(t["modality"] == "AB" for t in tract)
        rows.append(d)
    return rows


def opportunity_gaps(conn: sqlite3.Connection, efo_id: str, min_score: float = 0.3) -> list[dict]:
    """Well-evidenced, small-molecule-tractable targets with no clinical drug.

    This is the single most useful query in the package: it is the intersection
    of "biology says it matters", "chemistry says it is reachable", and
    "nobody has finished the job".
    """
    rows = target_landscape(conn, efo_id, limit=500)
    gaps = [
        r for r in rows
        if r["association_score"] and r["association_score"] >= min_score
        and r["small_molecule_tractable"]
        and (r["n_clinical_drugs"] or 0) == 0
    ]
    return sorted(gaps, key=lambda r: -r["association_score"])


def best_compounds(conn: sqlite3.Connection, symbol: str, limit: int = 20) -> list[dict]:
    """Most potent known compounds against a target, with druglikeness flags."""
    sql = """
    SELECT
        c.chembl_id, c.pref_name, c.smiles, c.max_phase,
        c.molecular_weight, c.alogp, c.ro5_violations,
        MAX(a.pchembl_value) AS best_pchembl,
        a.standard_type,
        COUNT(*)             AS n_measurements
    FROM activity a
    JOIN target   t ON t.chembl_id = a.target_chembl_id
    JOIN compound c ON c.chembl_id = a.chembl_id
    WHERE t.symbol = ?
    GROUP BY c.chembl_id
    ORDER BY best_pchembl DESC
    LIMIT ?
    """
    return [dict(r) for r in conn.execute(sql, (symbol, limit))]


def failure_patterns(conn: sqlite3.Connection, efo_id: str) -> dict:
    """Why trials in this indication stop. Read this before designing anything."""
    trials = [
        dict(r)
        for r in conn.execute(
            "SELECT t.* FROM trial t JOIN trial_disease td ON td.nct_id = t.nct_id "
            "WHERE td.efo_id = ?",
            (efo_id,),
        )
    ]
    stopped = [t for t in trials if t.get("why_stopped")]

    # Crude but genuinely useful bucketing of free-text stop reasons.
    buckets = {
        "accrual/recruitment": ["accrual", "recruit", "enroll"],
        "funding/business": ["fund", "business", "sponsor", "financial", "strategic"],
        "efficacy/futility": ["futility", "efficacy", "lack of response", "no benefit"],
        "safety/toxicity": ["toxicity", "safety", "adverse", "tolerab"],
        "investigator/administrative": ["investigator", "pi ", "administrative"],
    }
    tally: Counter = Counter()
    for t in stopped:
        reason = (t["why_stopped"] or "").lower()
        matched = False
        for label, keys in buckets.items():
            if any(k in reason for k in keys):
                tally[label] += 1
                matched = True
        if not matched:
            tally["other/unclear"] += 1

    return {
        "n_trials": len(trials),
        "by_status": dict(Counter(t["status"] for t in trials).most_common()),
        "by_phase": dict(Counter(t["phase"] or "N/A" for t in trials).most_common()),
        "n_stopped_with_reason": len(stopped),
        "stop_reason_buckets": dict(tally.most_common()),
        "examples": [
            {"nct_id": t["nct_id"], "phase": t["phase"], "reason": t["why_stopped"]}
            for t in stopped[:15]
        ],
    }


def format_landscape(rows: list[dict]) -> str:
    head = (
        f"{'symbol':<10}{'assoc':>7}{'actives':>9}{'best_pX':>9}"
        f"{'drugs':>7}{'phase':>7}  {'SM':<3}{'AB':<3} name"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['symbol']:<10}"
            f"{(r['association_score'] or 0):>7.3f}"
            f"{(r['n_actives'] or 0):>9}"
            f"{(r['best_pchembl'] or 0):>9.2f}"
            f"{(r['n_clinical_drugs'] or 0):>7}"
            f"{(r['highest_phase'] or 0):>7.0f}"
            f"  {'y' if r['small_molecule_tractable'] else '-':<3}"
            f"{'y' if r['antibody_tractable'] else '-':<3}"
            f" {(r['name'] or '')[:38]}"
        )
    return "\n".join(lines)
