"""ChEMBL -- the quantitative layer: what binds what, and how hard.

Open Targets tells you *which* target matters. ChEMBL tells you what chemical
matter exists against it and at what potency. This is where a target stops
being a hypothesis and starts being a drug discovery programme.

API: https://www.ebi.ac.uk/chembl/api/data/  (no key needed)
"""

from __future__ import annotations

from typing import Any, Iterator

from qc_research.db import now
from qc_research.http import request

BASE = "https://www.ebi.ac.uk/chembl/api/data"
SOURCE = "chembl"

_TYPE_RANK = {
    "SINGLE PROTEIN": 0,
    "PROTEIN COMPLEX": 1,
    "PROTEIN FAMILY": 2,
    "PROTEIN-PROTEIN INTERACTION": 3,
}


def _paginate(path: str, params: dict[str, Any], key: str, limit: int) -> Iterator[dict]:
    """Walk ChEMBL's offset pagination until `limit` records or exhaustion."""
    offset, fetched = 0, 0
    page_size = min(1000, limit)
    while fetched < limit:
        page = request(
            f"{BASE}/{path}.json",
            params={**params, "limit": page_size, "offset": offset},
        )
        records = page.get(key, [])
        if not records:
            return
        for rec in records:
            yield rec
            fetched += 1
            if fetched >= limit:
                return
        meta = page.get("page_meta", {})
        if not meta.get("next"):
            return
        offset += len(records)


def resolve_target(uniprot: str, organism: str = "Homo sapiens") -> str | None:
    """UniProt accession -> ChEMBL target ID, preferring SINGLE PROTEIN entries."""
    page = request(
        f"{BASE}/target.json",
        params={"target_components__accession": uniprot, "limit": 20},
    )
    candidates = [
        t for t in page.get("targets", [])
        if not organism or t.get("organism") == organism
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: _TYPE_RANK.get(t.get("target_type"), 99))
    return candidates[0]["target_chembl_id"]


def fetch_activities(
    target_chembl_id: str,
    limit: int = 500,
    min_pchembl: float | None = 5.0,
) -> list[dict]:
    """Measured potencies against a target.

    `pchembl_value` is -log10(molar potency), so 9 = 1 nM and 6 = 1 uM. It is
    the only field that is safely comparable across assay types, which is why
    we filter on it. Rows without one are usually unusable.
    """
    params: dict[str, Any] = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": "IC50,Ki,Kd,EC50",
        "pchembl_value__isnull": "false",
    }
    if min_pchembl is not None:
        params["pchembl_value__gte"] = min_pchembl

    ts = now()
    rows = []
    for a in _paginate("activity", params, "activities", limit):
        val = a.get("standard_value")
        pch = a.get("pchembl_value")
        rows.append(
            {
                "activity_id": str(a.get("activity_id")),
                "chembl_id": a.get("molecule_chembl_id"),
                "target_chembl_id": a.get("target_chembl_id"),
                "standard_type": a.get("standard_type"),
                "standard_value": float(val) if val is not None else None,
                "standard_units": a.get("standard_units"),
                "standard_relation": a.get("standard_relation"),
                "pchembl_value": float(pch) if pch is not None else None,
                "assay_chembl_id": a.get("assay_chembl_id"),
                "assay_type": a.get("assay_type"),
                "document_pmid": (
                    str(a["document_pmid"]) if a.get("document_pmid") else None
                ),
                "source": SOURCE,
                "fetched_at": ts,
            }
        )
    return rows


def fetch_compounds(chembl_ids: list[str], batch: int = 50) -> list[dict]:
    """Structures and physicochemical properties for a set of molecules."""
    ts = now()
    rows = []
    for i in range(0, len(chembl_ids), batch):
        chunk = chembl_ids[i : i + batch]
        page = request(
            f"{BASE}/molecule.json",
            params={"molecule_chembl_id__in": ",".join(chunk), "limit": batch},
        )
        for m in page.get("molecules", []):
            props = m.get("molecule_properties") or {}
            struct = m.get("molecule_structures") or {}
            max_phase = m.get("max_phase")
            rows.append(
                {
                    "chembl_id": m.get("molecule_chembl_id"),
                    "pref_name": m.get("pref_name"),
                    "smiles": struct.get("canonical_smiles"),
                    "inchi_key": struct.get("standard_inchi_key"),
                    "max_phase": float(max_phase) if max_phase is not None else None,
                    "molecular_weight": _f(props.get("full_mwt")),
                    "alogp": _f(props.get("alogp")),
                    "hbd": _i(props.get("hbd")),
                    "hba": _i(props.get("hba")),
                    "psa": _f(props.get("psa")),
                    "ro5_violations": _i(props.get("num_ro5_violations")),
                    "is_approved": int(bool(max_phase and float(max_phase) >= 4)),
                    "source": SOURCE,
                    "fetched_at": ts,
                }
            )
    return rows


def fetch_mechanisms(target_chembl_id: str, limit: int = 200) -> list[dict]:
    """Curated mechanism of action for drugs hitting this target.

    Sparse by design -- it only covers compounds that reached the clinic. That
    sparsity is informative: a target with zero mechanism rows has never had a
    drug succeed against it.
    """
    ts = now()
    rows = []
    for m in _paginate(
        "mechanism", {"target_chembl_id": target_chembl_id}, "mechanisms", limit
    ):
        rows.append(
            {
                "chembl_id": m.get("molecule_chembl_id"),
                "target_chembl_id": m.get("target_chembl_id"),
                "action_type": m.get("action_type"),
                "mechanism_of_action": m.get("mechanism_of_action"),
                "max_phase": _f(m.get("max_phase")),
                "source": SOURCE,
                "fetched_at": ts,
            }
        )
    return rows


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
