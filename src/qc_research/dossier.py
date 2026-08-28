"""Target deep-dive: everything the graph knows about one protein, in one place.

A dossier answers the question a project team actually asks before committing
people to a target: *should we work on this, and what will kill us?*

Four things decide that, and they come from four different sources:

    1. is the biology real?          Open Targets association + evidence types
    2. can we drug it?               tractability + existing chemical matter
    3. will it be toxic?             essentiality, genetic constraint, mouse KO
    4. has anyone tried?             mechanisms, clinical phase, trial outcomes

The assessment at the end is rule-based, not learned and not generated. Every
flag traces to a specific field, so you can argue with it.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field

from qc_research import db
from qc_research.sources import chembl, clinicaltrials, opentargets, pubmed

POTENT = 7.0
VERY_POTENT = 8.0


@dataclass
class Dossier:
    symbol: str
    ensembl_id: str = ""
    profile: dict = field(default_factory=dict)
    chemistry: dict = field(default_factory=dict)
    clinical: dict = field(default_factory=dict)
    literature: dict = field(default_factory=dict)
    focus_disease: dict | None = None
    assessment: dict = field(default_factory=dict)


def load_profile(conn: sqlite3.Connection, symbol: str, refresh: bool = False) -> dict:
    """Fetch (and cache in the graph) the Open Targets profile for a symbol."""
    row = conn.execute(
        "SELECT * FROM target WHERE symbol = ? COLLATE NOCASE", (symbol,)
    ).fetchone()

    if row is None:
        hits = opentargets._query(
            """query($q: String!) {
                 search(queryString: $q, entityNames: ["target"], page:{index:0,size:1}) {
                   hits { id name }
                 }
               }""",
            {"q": symbol},
        )["search"]["hits"]
        if not hits:
            raise ValueError(f"no target matched {symbol!r}")
        ensembl_id = hits[0]["id"]
    else:
        ensembl_id = row["ensembl_id"]

    if row is not None and row["profiled_at"] and not refresh:
        return _row_to_profile(row)

    p = opentargets.fetch_target_profile(ensembl_id)
    ts = db.now()
    db.upsert(
        conn,
        "target",
        [
            {
                "ensembl_id": p["ensembl_id"],
                "symbol": p["symbol"],
                "name": p["name"],
                "biotype": p["biotype"],
                "uniprot_ids": json.dumps(p["uniprot_ids"]),
                "chembl_id": (row["chembl_id"] if row else None),
                "tractability": json.dumps(p["tractability"]),
                "function": p["function"],
                "target_class": json.dumps(p["target_class"]),
                "subcellular_locations": json.dumps(p["subcellular_locations"]),
                "pathways": json.dumps(p["pathways"]),
                "is_essential": (
                    None if p["is_essential"] is None else int(p["is_essential"])
                ),
                "loeuf_score": p["loeuf_score"],
                "loeuf_bin": p["loeuf_bin"],
                "safety_liabilities": json.dumps(p["safety_liabilities"]),
                "chemical_probes": json.dumps(p["chemical_probes"]),
                "mouse_phenotypes": json.dumps(p["mouse_phenotypes"]),
                "n_associated_diseases": p["n_associated_diseases"],
                "profiled_at": ts,
                "source": opentargets.SOURCE,
                "fetched_at": ts,
            }
        ],
    )
    p["associated_diseases_full"] = p["associated_diseases"]
    return p


def _row_to_profile(row: sqlite3.Row) -> dict:
    j = lambda k: json.loads(row[k] or "[]")
    return {
        "ensembl_id": row["ensembl_id"],
        "symbol": row["symbol"],
        "name": row["name"],
        "biotype": row["biotype"],
        "function": row["function"],
        "target_class": j("target_class"),
        "uniprot_ids": j("uniprot_ids"),
        "subcellular_locations": j("subcellular_locations"),
        "pathways": j("pathways"),
        "tractability": j("tractability"),
        "is_essential": (
            None if row["is_essential"] is None else bool(row["is_essential"])
        ),
        "loeuf_score": row["loeuf_score"],
        "loeuf_bin": row["loeuf_bin"],
        "safety_liabilities": j("safety_liabilities"),
        "chemical_probes": j("chemical_probes"),
        "mouse_phenotypes": j("mouse_phenotypes"),
        "n_associated_diseases": row["n_associated_diseases"],
        "associated_diseases": [],
        "chembl_id": row["chembl_id"],
    }


def chemical_matter(
    conn: sqlite3.Connection, symbol: str, chembl_target: str | None, fetch: bool = True
) -> dict:
    """Potency distribution and druglikeness of everything known to bind."""
    if chembl_target is None:
        row = conn.execute(
            "SELECT chembl_id FROM target WHERE symbol = ? COLLATE NOCASE", (symbol,)
        ).fetchone()
        chembl_target = row["chembl_id"] if row else None

    if chembl_target is None:
        return {"chembl_target": None, "n_compounds": 0, "note": "no ChEMBL target"}

    have = conn.execute(
        "SELECT COUNT(*) AS n FROM activity WHERE target_chembl_id = ?",
        (chembl_target,),
    ).fetchone()["n"]

    if have == 0 and fetch:
        acts = chembl.fetch_activities(chembl_target, limit=1000, min_pchembl=5.0)
        db.upsert(conn, "activity", acts)
        ids = sorted({a["chembl_id"] for a in acts if a["chembl_id"]})
        db.upsert(conn, "compound", chembl.fetch_compounds(ids))

    rows = [
        dict(r)
        for r in conn.execute(
            """SELECT c.chembl_id, c.pref_name, c.smiles, c.max_phase,
                      c.molecular_weight, c.alogp, c.psa, c.ro5_violations,
                      MAX(a.pchembl_value) AS best_pchembl,
                      COUNT(*)             AS n_measurements
               FROM activity a
               JOIN compound c ON c.chembl_id = a.chembl_id
               WHERE a.target_chembl_id = ?
               GROUP BY c.chembl_id
               ORDER BY best_pchembl DESC""",
            (chembl_target,),
        )
    ]

    potencies = [r["best_pchembl"] for r in rows if r["best_pchembl"] is not None]
    ro5_ok = [r for r in rows if (r["ro5_violations"] or 0) <= 1]

    types = {
        r["standard_type"]: r["n"]
        for r in conn.execute(
            "SELECT standard_type, COUNT(*) AS n FROM activity "
            "WHERE target_chembl_id = ? GROUP BY standard_type",
            (chembl_target,),
        )
    }

    return {
        "chembl_target": chembl_target,
        "n_compounds": len(rows),
        "n_measurements": sum(types.values()),
        "assay_types": types,
        "potency": {
            "best": max(potencies) if potencies else None,
            "median": statistics.median(potencies) if potencies else None,
            "n_potent": sum(1 for p in potencies if p >= POTENT),
            "n_very_potent": sum(1 for p in potencies if p >= VERY_POTENT),
        },
        "druglikeness": {
            "n_ro5_compliant": len(ro5_ok),
            "pct_ro5_compliant": (100 * len(ro5_ok) / len(rows)) if rows else 0.0,
            "median_mw": (
                statistics.median(
                    [r["molecular_weight"] for r in rows if r["molecular_weight"]]
                )
                if any(r["molecular_weight"] for r in rows)
                else None
            ),
        },
        "top_compounds": rows[:10],
    }


def clinical_precedent(
    conn: sqlite3.Connection, chembl_target: str | None, disease_name: str | None
) -> dict:
    """Drugs with a curated mechanism against this target, plus their trials."""
    if chembl_target is None:
        return {"n_drugs": 0, "max_phase": None, "drugs": [], "trials": {}}

    if conn.execute(
        "SELECT COUNT(*) AS n FROM mechanism WHERE target_chembl_id = ?",
        (chembl_target,),
    ).fetchone()["n"] == 0:
        db.upsert(conn, "mechanism", chembl.fetch_mechanisms(chembl_target))

    drugs = [
        dict(r)
        for r in conn.execute(
            """SELECT m.chembl_id, m.action_type, m.mechanism_of_action, m.max_phase,
                      c.pref_name, c.smiles
               FROM mechanism m
               LEFT JOIN compound c ON c.chembl_id = m.chembl_id
               WHERE m.target_chembl_id = ?
               ORDER BY m.max_phase DESC NULLS LAST""",
            (chembl_target,),
        )
    ]

    missing = [d["chembl_id"] for d in drugs if not d["pref_name"] and d["chembl_id"]]
    if missing:
        db.upsert(conn, "compound", chembl.fetch_compounds(missing))
        for d in drugs:
            if d["chembl_id"] in missing:
                r = conn.execute(
                    "SELECT pref_name, smiles FROM compound WHERE chembl_id = ?",
                    (d["chembl_id"],),
                ).fetchone()
                if r:
                    d["pref_name"], d["smiles"] = r["pref_name"], r["smiles"]

    phases = [d["max_phase"] for d in drugs if d["max_phase"] is not None]

    trial_summary: dict = {}
    named = [d["pref_name"] for d in drugs if d["pref_name"]][:5]
    if named and disease_name:
        found = []
        for name in named:
            found += clinicaltrials.fetch_trials(disease_name, limit=40, intervention=name)
        if found:
            db.upsert(conn, "trial", found)
            trial_summary = clinicaltrials.failure_analysis(found)

    return {
        "n_drugs": len(drugs),
        "max_phase": max(phases) if phases else None,
        "note": (
            "no curated mechanism in ChEMBL -- check compound max_phase above, "
            "which covers clinical-stage binders without a curated MoA"
            if not drugs
            else None
        ),
        "n_approved": sum(1 for p in phases if p >= 4),
        "action_types": sorted({d["action_type"] for d in drugs if d["action_type"]}),
        "drugs": drugs[:15],
        "trials": trial_summary,
    }


def literature(
    conn: sqlite3.Connection, symbol: str, disease_name: str | None, n: int = 15
) -> dict:
    """Reviews plus a publication-count trend, which reveals momentum."""
    if disease_name:
        query = pubmed.target_disease_query(symbol, disease_name, review_only=True)
    else:
        query = f'("{symbol}"[Title/Abstract] OR "{symbol}"[Gene Name]) AND review[Publication Type]'

    pmids = pubmed.search(query, retmax=n)
    papers = pubmed.fetch_articles(pmids) if pmids else []
    if papers:
        db.upsert(conn, "paper", papers)
        db.upsert(
            conn,
            "paper_mention",
            [
                {
                    "pmid": p["pmid"],
                    "entity_type": "target",
                    "entity_id": symbol,
                    "context": query[:200],
                }
                for p in papers
            ],
        )

    from collections import Counter

    return {
        "query": query,
        "n_reviews": len(papers),
        "by_year": dict(sorted(Counter(p["year"] for p in papers if p["year"]).items())),
        "key_reviews": [
            {
                "pmid": p["pmid"],
                "year": p["year"],
                "journal": p["journal"],
                "title": p["title"],
            }
            for p in sorted(papers, key=lambda x: -(x["year"] or 0))[:8]
        ],
    }


def assess(d: Dossier) -> dict:
    """Rule-based flags. Every one names the field that produced it."""
    p, chem, clin = d.profile, d.chemistry, d.clinical
    strengths: list[str] = []
    risks: list[str] = []

    modalities = {t["modality"] for t in p.get("tractability", [])}
    if "SM" in modalities:
        strengths.append("small-molecule tractable (Open Targets tractability)")
    else:
        risks.append("no small-molecule tractability evidence -- may need a biologic")

    score = (d.focus_disease or {}).get("score")
    if score is not None:
        if score >= 0.5:
            strengths.append(f"strong disease association (score {score:.2f})")
        elif score < 0.2:
            risks.append(f"weak disease association (score {score:.2f})")

    n_compounds = chem.get("n_compounds", 0)
    n_potent = (chem.get("potency") or {}).get("n_potent") or 0
    if n_potent >= 50:
        strengths.append(f"mature chemical matter ({n_potent} compounds at pChEMBL>={POTENT:.0f})")
    elif n_compounds == 0:
        risks.append("no measured chemical matter -- hit finding starts from zero")
    elif n_potent < 5:
        risks.append(
            f"thin chemical matter: {n_compounds} compound(s) measured, "
            f"{n_potent} at pChEMBL>={POTENT:.0f}"
        )

    probes = [x for x in p.get("chemical_probes", []) if x.get("high_quality")]
    if probes:
        strengths.append(f"{len(probes)} high-quality chemical probe(s) available")
    else:
        risks.append("no high-quality chemical probe -- target validation will be harder")

    compound_phase = max(
        (c["max_phase"] or 0) for c in (chem.get("top_compounds") or [{"max_phase": 0}])
    )
    best_phase = max(clin.get("max_phase") or 0, compound_phase)

    if clin.get("n_approved"):
        strengths.append(f"clinically precedented ({clin['n_approved']} approved drug(s))")
    elif best_phase >= 4:
        strengths.append("an approved drug exists among known binders")
    elif best_phase >= 1:
        strengths.append(f"reached phase {best_phase:.0f} in the clinic")
    else:
        risks.append("no compound against this target has entered the clinic")

    if p.get("is_essential"):
        risks.append("pan-essential in DepMap -- systemic inhibition likely toxic")
    if p.get("loeuf_bin") == 0:
        risks.append(
            "most-constrained LOEUF decile -- human loss-of-function is depleted, "
            "suggesting low tolerance for inhibition"
        )
    if p.get("safety_liabilities"):
        risks.append(f"{len(p['safety_liabilities'])} recorded safety liability event(s)")

    n_dis = p.get("n_associated_diseases") or 0
    if n_dis > 1000:
        risks.append(
            f"associated with {n_dis} diseases -- highly pleiotropic, selectivity will be hard"
        )

    tractable = "SM" in modalities

    if clin.get("n_approved") or best_phase >= 4:
        verdict = "precedented -- competition, not feasibility, is the question"
    elif not tractable:
        alt = sorted(modalities - {"SM"})
        verdict = (
            "difficult -- no small-molecule tractability"
            + (f"; consider {'/'.join(alt)}" if alt else "")
        )
    elif n_potent >= 20:
        stage = f", best phase {best_phase:.0f}" if best_phase else ""
        verdict = (
            "chemically enabled but not yet approved"
            f"{stage} -- the interesting quadrant"
        )
    elif n_compounds == 0:
        verdict = "tractable in principle, nothing made yet -- highest risk, highest novelty"
    else:
        verdict = (
            f"tractable, but chemical matter is early "
            f"({n_compounds} compounds, {n_potent} potent) -- needs hit finding"
        )

    return {"verdict": verdict, "strengths": strengths, "risks": risks}


def build(
    conn: sqlite3.Connection,
    symbol: str,
    disease: str | None = None,
    refresh: bool = False,
    skip_literature: bool = False,
) -> Dossier:
    profile = load_profile(conn, symbol, refresh=refresh)
    d = Dossier(symbol=profile["symbol"], ensembl_id=profile["ensembl_id"], profile=profile)

    disease_name = None
    if disease:
        row = conn.execute(
            "SELECT d.efo_id, d.name, dt.score, dt.datatype_scores "
            "FROM disease d JOIN disease_target dt ON dt.efo_id = d.efo_id "
            "WHERE d.name LIKE ? AND dt.ensembl_id = ?",
            (f"%{disease}%", d.ensembl_id),
        ).fetchone()
        if row:
            d.focus_disease = {
                "efo_id": row["efo_id"],
                "name": row["name"],
                "score": row["score"],
                "datatype_scores": json.loads(row["datatype_scores"] or "{}"),
            }
            disease_name = row["name"]
        else:
            associations = profile.get("associated_diseases") or []
            if not associations:
                associations = opentargets.fetch_target_profile(
                    d.ensembl_id, n_diseases=200
                )["associated_diseases"]
                profile["associated_diseases"] = associations

            needle = disease.lower()
            match = next(
                (x for x in associations if needle in x["name"].lower()), None
            )
            if match:
                d.focus_disease = match
                disease_name = match["name"]
            else:
                disease_name = disease
                d.focus_disease = {
                    "efo_id": None,
                    "name": disease,
                    "score": None,
                    "datatype_scores": {},
                    "note": (
                        f"{disease!r} is not among the top associated diseases for "
                        f"{d.symbol} -- used as a free-text filter only"
                    ),
                }

    chembl_target = profile.get("chembl_id")
    if not chembl_target and profile.get("uniprot_ids"):
        chembl_target = chembl.resolve_target(profile["uniprot_ids"][0])
        if chembl_target:
            conn.execute(
                "UPDATE target SET chembl_id = ? WHERE ensembl_id = ?",
                (chembl_target, d.ensembl_id),
            )
            conn.commit()

    d.chemistry = chemical_matter(conn, d.symbol, chembl_target)
    d.clinical = clinical_precedent(conn, chembl_target, disease_name)
    if not skip_literature:
        d.literature = literature(conn, d.symbol, disease_name)
    d.assessment = assess(d)
    return d
