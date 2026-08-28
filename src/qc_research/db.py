"""The knowledge graph, as plain SQL.

Design notes:

  * Nodes are typed tables (disease, target, compound, trial, paper) rather
    than a generic node/edge blob. Typed columns catch bad data early and
    make the queries readable.
  * Every row carries `source` and `fetched_at`. Provenance is not optional:
    the moment you cannot say where a number came from, the graph is a
    liability rather than an asset.
  * SQLite is deliberate. Zero install, one file, and at this scale
    (10^5-10^6 rows) it is entirely adequate. Swapping to DuckDB or Postgres
    later is a driver change, not a rewrite.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DB = Path("data/qc.sqlite")

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------- nodes ----------

CREATE TABLE IF NOT EXISTS disease (
    efo_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    therapeutic_areas TEXT,       -- JSON array
    source       TEXT NOT NULL,
    fetched_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS target (
    ensembl_id       TEXT PRIMARY KEY,
    symbol           TEXT NOT NULL,
    name             TEXT,
    biotype          TEXT,
    uniprot_ids      TEXT,        -- JSON array
    chembl_id        TEXT,
    tractability     TEXT,        -- JSON: is it druggable at all?
    -- profile columns, populated by `dossier` (Open Targets target endpoint)
    function              TEXT,
    target_class          TEXT,   -- JSON array
    subcellular_locations TEXT,   -- JSON array
    pathways              TEXT,   -- JSON array
    is_essential          INTEGER,-- pan-essential in DepMap cell lines
    loeuf_score           REAL,   -- loss-of-function observed/expected
    loeuf_bin             INTEGER,-- 0 = most constrained decile of human genes
    safety_liabilities    TEXT,   -- JSON array
    chemical_probes       TEXT,   -- JSON array
    mouse_phenotypes      TEXT,   -- JSON array
    n_associated_diseases INTEGER,
    profiled_at           TEXT,
    source           TEXT NOT NULL,
    fetched_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_target_symbol ON target(symbol);
CREATE INDEX IF NOT EXISTS idx_target_chembl ON target(chembl_id);

CREATE TABLE IF NOT EXISTS compound (
    chembl_id        TEXT PRIMARY KEY,
    pref_name        TEXT,
    smiles           TEXT,
    inchi_key        TEXT,
    max_phase        REAL,        -- 4 = approved drug, 0/NULL = preclinical
    molecular_weight REAL,
    alogp            REAL,
    hbd              INTEGER,
    hba              INTEGER,
    psa              REAL,
    ro5_violations   INTEGER,
    is_approved      INTEGER,
    source           TEXT NOT NULL,
    fetched_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compound_phase ON compound(max_phase);

CREATE TABLE IF NOT EXISTS trial (
    nct_id           TEXT PRIMARY KEY,
    title            TEXT,
    status           TEXT,
    phase            TEXT,
    enrollment       INTEGER,
    start_date       TEXT,
    completion_date  TEXT,
    why_stopped      TEXT,        -- the single most informative field here
    sponsor          TEXT,
    conditions       TEXT,        -- JSON array
    interventions    TEXT,        -- JSON array
    primary_outcomes TEXT,        -- JSON array
    has_results      INTEGER,
    source           TEXT NOT NULL,
    fetched_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trial_status ON trial(status);
CREATE INDEX IF NOT EXISTS idx_trial_phase ON trial(phase);

CREATE TABLE IF NOT EXISTS paper (
    pmid         TEXT PRIMARY KEY,
    doi          TEXT,
    title        TEXT,
    abstract     TEXT,
    journal      TEXT,
    year         INTEGER,
    authors      TEXT,            -- JSON array
    mesh_terms   TEXT,            -- JSON array
    pub_types    TEXT,            -- JSON array
    source       TEXT NOT NULL,
    fetched_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_paper_year ON paper(year);

-- ---------- edges ----------

-- How strongly is this target implicated in this disease, and by what evidence?
CREATE TABLE IF NOT EXISTS disease_target (
    efo_id       TEXT NOT NULL,
    ensembl_id   TEXT NOT NULL,
    score        REAL,
    datatype_scores TEXT,         -- JSON: genetic / known drug / literature / ...
    source       TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (efo_id, ensembl_id)
);

-- Measured activity of a compound against a target. The quantitative core.
CREATE TABLE IF NOT EXISTS activity (
    activity_id      TEXT PRIMARY KEY,
    chembl_id        TEXT NOT NULL,
    target_chembl_id TEXT NOT NULL,
    standard_type    TEXT,        -- IC50, Ki, EC50, Kd
    standard_value   REAL,
    standard_units   TEXT,
    standard_relation TEXT,       -- '=', '>', '<' -- censored data matters
    pchembl_value    REAL,        -- -log10(molar); the comparable number
    assay_chembl_id  TEXT,
    assay_type       TEXT,
    document_pmid    TEXT,
    source           TEXT NOT NULL,
    fetched_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activity_target ON activity(target_chembl_id);
CREATE INDEX IF NOT EXISTS idx_activity_compound ON activity(chembl_id);
CREATE INDEX IF NOT EXISTS idx_activity_pchembl ON activity(pchembl_value);

-- Curated mechanism of action: how an approved/clinical drug acts on a target.
CREATE TABLE IF NOT EXISTS mechanism (
    chembl_id        TEXT NOT NULL,
    target_chembl_id TEXT,
    action_type      TEXT,        -- INHIBITOR, AGONIST, ...
    mechanism_of_action TEXT,
    max_phase        REAL,
    source           TEXT NOT NULL,
    fetched_at       TEXT NOT NULL,
    PRIMARY KEY (chembl_id, target_chembl_id, mechanism_of_action)
);

CREATE TABLE IF NOT EXISTS trial_disease (
    nct_id   TEXT NOT NULL,
    efo_id   TEXT NOT NULL,
    PRIMARY KEY (nct_id, efo_id)
);

CREATE TABLE IF NOT EXISTS paper_mention (
    pmid        TEXT NOT NULL,
    entity_type TEXT NOT NULL,    -- 'target' | 'disease' | 'compound'
    entity_id   TEXT NOT NULL,
    context     TEXT,             -- why we think it is mentioned
    PRIMARY KEY (pmid, entity_type, entity_id)
);

-- ---------- provenance ----------

CREATE TABLE IF NOT EXISTS ingest_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    operation   TEXT NOT NULL,
    query       TEXT,
    n_rows      INTEGER,
    started_at  TEXT,
    finished_at TEXT,
    error       TEXT
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "target": [
        ("function", "TEXT"),
        ("target_class", "TEXT"),
        ("subcellular_locations", "TEXT"),
        ("pathways", "TEXT"),
        ("is_essential", "INTEGER"),
        ("loeuf_score", "REAL"),
        ("loeuf_bin", "INTEGER"),
        ("safety_liabilities", "TEXT"),
        ("chemical_probes", "TEXT"),
        ("mouse_phenotypes", "TEXT"),
        ("n_associated_diseases", "INTEGER"),
        ("profiled_at", "TEXT"),
    ],
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        existing = {
            r["name"] for r in conn.execute(f"PRAGMA table_info({table})")
        }
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def upsert(
    conn: sqlite3.Connection,
    table: str,
    rows: Iterable[dict[str, Any]],
) -> int:
    """Insert-or-replace a batch. Returns the number of rows written."""
    rows = list(rows)
    if not rows:
        return 0

    cols = list(rows[0].keys())
    placeholders = ", ".join("?" * len(cols))
    sql = (
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    conn.commit()
    return len(rows)


def log_ingest(
    conn: sqlite3.Connection,
    source: str,
    operation: str,
    query: str,
    n_rows: int,
    started_at: str,
    error: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, operation, query, n_rows, started_at, "
        "finished_at, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source, operation, query, n_rows, started_at, now(), error),
    )
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    tables = [
        "disease", "target", "compound", "trial", "paper",
        "disease_target", "activity", "mechanism", "paper_mention",
    ]
    return {
        t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
        for t in tables
    }
