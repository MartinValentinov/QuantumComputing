"""Command line entry point.

    python -m qc_research ingest "pancreatic ductal adenocarcinoma"
    python -m qc_research landscape "pancreatic ductal adenocarcinoma"
    python -m qc_research gaps      "pancreatic ductal adenocarcinoma"
    python -m qc_research failures  "pancreatic ductal adenocarcinoma"
    python -m qc_research compounds KRAS
    python -m qc_research dossier   KRAS --disease "pancreatic ductal adenocarcinoma"
    python -m qc_research stats
"""

from __future__ import annotations

import argparse
import json
import sys

from qc_research import db, dossier, queries, render
from qc_research.pipeline import IngestConfig, ingest_disease
from qc_research.sources import opentargets


def _resolve(conn, name: str) -> tuple[str, str]:
    """Prefer a disease already in the local graph; fall back to the API."""
    row = conn.execute(
        "SELECT efo_id, name FROM disease WHERE lower(name) LIKE ?",
        (f"%{name.lower()}%",),
    ).fetchone()
    if row:
        return row["efo_id"], row["name"]
    hits = opentargets.search_disease(name, size=1)
    if not hits:
        sys.exit(f"no disease matched {name!r}")
    return hits[0]["efo_id"], hits[0]["name"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qc_research", description=__doc__)
    ap.add_argument("--db", default=str(db.DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="build the graph for a disease")
    p.add_argument("disease")
    p.add_argument("--targets", type=int, default=25)
    p.add_argument("--activities", type=int, default=300)
    p.add_argument("--min-pchembl", type=float, default=6.0)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--no-literature", action="store_true")

    for name, help_text in [
        ("landscape", "target triage table"),
        ("gaps", "evidenced + tractable + undrugged targets"),
        ("failures", "why trials stopped in this indication"),
    ]:
        q = sub.add_parser(name, help=help_text)
        q.add_argument("disease")
        q.add_argument("--json", action="store_true")

    c = sub.add_parser("compounds", help="best known compounds for a target")
    c.add_argument("symbol")
    c.add_argument("--limit", type=int, default=20)
    c.add_argument("--json", action="store_true")

    ds = sub.add_parser("dossier", help="deep dive on a single target")
    ds.add_argument("symbol", help="gene symbol, e.g. KRAS")
    ds.add_argument("--disease", default=None, help="focus indication")
    ds.add_argument("--refresh", action="store_true", help="re-fetch the profile")
    ds.add_argument("--no-literature", action="store_true")
    ds.add_argument("--markdown", action="store_true")
    ds.add_argument("--json", action="store_true")
    ds.add_argument("--out", default=None, help="write to a file instead of stdout")

    sub.add_parser("stats", help="row counts per table")

    args = ap.parse_args(argv)
    conn = db.connect(args.db)

    if args.cmd == "ingest":
        cfg = IngestConfig(
            n_targets=args.targets,
            activities_per_target=args.activities,
            min_pchembl=args.min_pchembl,
            n_trials=args.trials,
            fetch_literature=not args.no_literature,
        )
        report = ingest_disease(conn, args.disease, cfg)
        print()
        print(report.render())

    elif args.cmd == "landscape":
        efo_id, name = _resolve(conn, args.disease)
        rows = queries.target_landscape(conn, efo_id)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{name}  [{efo_id}]\n")
            print(queries.format_landscape(rows))

    elif args.cmd == "gaps":
        efo_id, name = _resolve(conn, args.disease)
        rows = queries.opportunity_gaps(conn, efo_id)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{name}: {len(rows)} evidenced, tractable, undrugged targets\n")
            print(queries.format_landscape(rows))

    elif args.cmd == "failures":
        efo_id, name = _resolve(conn, args.disease)
        print(json.dumps(queries.failure_patterns(conn, efo_id), indent=2))

    elif args.cmd == "compounds":
        rows = queries.best_compounds(conn, args.symbol, args.limit)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{'chembl_id':<16}{'pX':>6}{'phase':>7}{'MW':>8}{'logP':>7}  smiles")
            for r in rows:
                print(
                    f"{r['chembl_id']:<16}"
                    f"{(r['best_pchembl'] or 0):>6.2f}"
                    f"{(r['max_phase'] or 0):>7.0f}"
                    f"{(r['molecular_weight'] or 0):>8.1f}"
                    f"{(r['alogp'] or 0):>7.2f}  {(r['smiles'] or '')[:48]}"
                )

    elif args.cmd == "dossier":
        d = dossier.build(
            conn,
            args.symbol,
            disease=args.disease,
            refresh=args.refresh,
            skip_literature=args.no_literature,
        )
        if args.json:
            from dataclasses import asdict

            text = json.dumps(asdict(d), indent=2, default=str)
        elif args.markdown:
            text = render.dossier_markdown(d)
        else:
            text = render.dossier_text(d)

        if args.out:
            from pathlib import Path

            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text)

    elif args.cmd == "stats":
        for t, n in db.stats(conn).items():
            print(f"{t:<18}{n:>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
