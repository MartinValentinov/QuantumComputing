"""Orchestration: one disease name in, a populated knowledge graph out.

The flow, and why it is ordered this way:

    disease name
      -> resolve to an ontology ID            (names are ambiguous, IDs are not)
      -> top associated targets               (Open Targets, evidence-weighted)
      -> map each target to ChEMBL            (via UniProt accession)
      -> known actives + potencies            (is there chemical matter at all?)
      -> curated mechanisms                   (has anything ever reached clinic?)
      -> clinical trials for the disease      (what was tried, what failed, why)
      -> review literature per target         (the narrative context)

Every stage is resumable: HTTP responses are cached on disk and writes are
INSERT OR REPLACE, so re-running is cheap and idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from qc_research import db
from qc_research.sources import chembl, clinicaltrials, opentargets, pubmed


@dataclass
class IngestConfig:
    n_targets: int = 25
    activities_per_target: int = 300
    min_pchembl: float = 6.0
    n_trials: int = 300
    papers_per_target: int = 10
    reviews_only: bool = True
    fetch_literature: bool = True


@dataclass
class IngestReport:
    disease: str = ""
    efo_id: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    targets_resolved: int = 0
    targets_unmapped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"{self.disease}  [{self.efo_id}]",
            "=" * (len(self.disease) + len(self.efo_id) + 4),
            "",
            "graph contents:",
        ]
        for table, n in self.counts.items():
            lines.append(f"  {table:<18} {n:>7}")
        lines.append("")
        lines.append(
            f"targets mapped to ChEMBL: {self.targets_resolved}"
            f" (unmapped: {len(self.targets_unmapped)})"
        )
        if self.targets_unmapped:
            lines.append(f"  no ChEMBL entry: {', '.join(self.targets_unmapped[:12])}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def ingest_disease(
    conn: sqlite3.Connection,
    disease_query: str,
    config: IngestConfig | None = None,
    verbose: bool = True,
) -> IngestReport:
    cfg = config or IngestConfig()
    report = IngestReport()

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    hits = opentargets.search_disease(disease_query, size=3)
    if not hits:
        raise ValueError(f"no disease matched {disease_query!r}")
    efo_id, name = hits[0]["efo_id"], hits[0]["name"]
    report.disease, report.efo_id = name, efo_id
    say(f"[1/6] resolved {disease_query!r} -> {name} ({efo_id})")
    if len(hits) > 1:
        alts = ", ".join(f"{h['name']} ({h['efo_id']})" for h in hits[1:])
        report.warnings.append(f"other candidate diseases: {alts}")

    started = db.now()
    ot = opentargets.fetch_disease_targets(efo_id, size=cfg.n_targets)
    db.upsert(conn, "disease", ot["disease"])
    db.upsert(conn, "target", ot["target"])
    db.upsert(conn, "disease_target", ot["disease_target"])
    db.log_ingest(
        conn, "opentargets", "disease_targets", efo_id, len(ot["target"]), started
    )
    say(
        f"[2/6] {len(ot['target'])} targets "
        f"(of {ot['_total_associated']} associated)"
    )

    say(f"[3/6] mapping targets to ChEMBL and pulling bioactivity...")
    all_compound_ids: set[str] = set()
    for i, t in enumerate(ot["target"], 1):
        uniprots = json.loads(t["uniprot_ids"] or "[]")
        chembl_target = None
        for acc in uniprots[:1]:
            chembl_target = chembl.resolve_target(acc)
            if chembl_target:
                break

        if not chembl_target:
            report.targets_unmapped.append(t["symbol"])
            continue

        conn.execute(
            "UPDATE target SET chembl_id = ? WHERE ensembl_id = ?",
            (chembl_target, t["ensembl_id"]),
        )
        report.targets_resolved += 1

        acts = chembl.fetch_activities(
            chembl_target,
            limit=cfg.activities_per_target,
            min_pchembl=cfg.min_pchembl,
        )
        db.upsert(conn, "activity", acts)
        all_compound_ids.update(a["chembl_id"] for a in acts if a["chembl_id"])

        mechs = chembl.fetch_mechanisms(chembl_target)
        db.upsert(conn, "mechanism", mechs)
        all_compound_ids.update(m["chembl_id"] for m in mechs if m["chembl_id"])

        say(
            f"       {i:>3}/{len(ot['target'])} {t['symbol']:<10} "
            f"{chembl_target:<16} {len(acts):>4} activities, "
            f"{len(mechs):>3} mechanisms"
        )
    conn.commit()

    say(f"[4/6] fetching {len(all_compound_ids)} compound structures...")
    compounds = chembl.fetch_compounds(sorted(all_compound_ids))
    db.upsert(conn, "compound", compounds)

    say("[5/6] fetching clinical trials (including terminated)...")
    started = db.now()
    trials = clinicaltrials.fetch_trials(name, limit=cfg.n_trials)
    db.upsert(conn, "trial", trials)
    db.upsert(
        conn,
        "trial_disease",
        [{"nct_id": t["nct_id"], "efo_id": efo_id} for t in trials if t["nct_id"]],
    )
    db.log_ingest(
        conn, "clinicaltrials.gov", "trials", name, len(trials), started
    )
    say(f"       {len(trials)} trials")

    if cfg.fetch_literature:
        say("[6/6] fetching review literature per target...")
        mapped = conn.execute(
            "SELECT symbol FROM target WHERE chembl_id IS NOT NULL"
        ).fetchall()
        for row in mapped:
            symbol = row["symbol"]
            q = pubmed.target_disease_query(symbol, name, review_only=cfg.reviews_only)
            pmids = pubmed.search(q, retmax=cfg.papers_per_target)
            if not pmids:
                continue
            papers = pubmed.fetch_articles(pmids)
            db.upsert(conn, "paper", papers)
            db.upsert(
                conn,
                "paper_mention",
                [
                    {
                        "pmid": p["pmid"],
                        "entity_type": "target",
                        "entity_id": symbol,
                        "context": "pubmed query match",
                    }
                    for p in papers
                ],
            )
    else:
        say("[6/6] literature skipped")

    report.counts = db.stats(conn)
    return report
