"""Open Targets Platform -- disease/target associations with evidence breakdown.

Why this is the entry point of the whole pipeline: Open Targets has already
done the hard work of integrating genetics, expression, pathways, known drugs
and literature into a single association score per (disease, target) pair.
Starting anywhere else means rebuilding that.

API: https://api.platform.opentargets.org/api/v4/graphql  (no key needed)
"""

from __future__ import annotations

import json
from typing import Any

from qc_research.db import now
from qc_research.http import request

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
SOURCE = "opentargets"


def _query(gql: str, variables: dict[str, Any]) -> dict:
    payload = request(ENDPOINT, json_body={"query": gql, "variables": variables})
    if "errors" in payload:
        raise RuntimeError(f"Open Targets GraphQL error: {payload['errors']}")
    return payload["data"]


SEARCH_DISEASE = """
query SearchDisease($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: $size}) {
    hits { id name description }
  }
}
"""


def search_disease(query: str, size: int = 5) -> list[dict]:
    """Resolve a free-text disease name to ontology IDs. Always do this first --
    'lung cancer' is not an identifier, MONDO_0008903 is."""
    data = _query(SEARCH_DISEASE, {"q": query, "size": size})
    return [
        {"efo_id": h["id"], "name": h["name"], "description": h.get("description")}
        for h in data["search"]["hits"]
    ]


DISEASE_TARGETS = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    description
    therapeuticAreas { id name }
    associatedTargets(page: {index: 0, size: $size}) {
      count
      rows {
        score
        datatypeScores { id score }
        target {
          id
          approvedSymbol
          approvedName
          biotype
          proteinIds { id source }
          dbXrefs { id source }
          tractability { label modality value }
        }
      }
    }
  }
}
"""


def fetch_disease_targets(efo_id: str, size: int = 50) -> dict[str, list[dict]]:
    """Top `size` targets associated with a disease, ranked by overall score.

    Returns rows for three tables: disease, target, disease_target.
    """
    data = _query(DISEASE_TARGETS, {"efoId": efo_id, "size": size})
    d = data.get("disease")
    if d is None:
        raise ValueError(f"Open Targets has no disease {efo_id!r}")

    ts = now()
    disease_row = {
        "efo_id": d["id"],
        "name": d["name"],
        "description": d.get("description"),
        "therapeutic_areas": json.dumps(
            [ta["name"] for ta in (d.get("therapeuticAreas") or [])]
        ),
        "source": SOURCE,
        "fetched_at": ts,
    }

    targets, edges = [], []
    for row in d["associatedTargets"]["rows"]:
        t = row["target"]

        uniprots = [
            p["id"] for p in (t.get("proteinIds") or [])
            if p.get("source") in ("uniprot_swissprot", "uniprot_trembl")
        ]
        chembl = next(
            (x["id"] for x in (t.get("dbXrefs") or []) if x.get("source") == "chEMBL"),
            None,
        )
        tract = [
            {"modality": x["modality"], "label": x["label"]}
            for x in (t.get("tractability") or [])
            if x.get("value")
        ]

        targets.append(
            {
                "ensembl_id": t["id"],
                "symbol": t["approvedSymbol"],
                "name": t.get("approvedName"),
                "biotype": t.get("biotype"),
                "uniprot_ids": json.dumps(uniprots),
                "chembl_id": chembl,
                "tractability": json.dumps(tract),
                "source": SOURCE,
                "fetched_at": ts,
            }
        )
        edges.append(
            {
                "efo_id": d["id"],
                "ensembl_id": t["id"],
                "score": row["score"],
                "datatype_scores": json.dumps(
                    {s["id"]: s["score"] for s in (row.get("datatypeScores") or [])}
                ),
                "source": SOURCE,
                "fetched_at": ts,
            }
        )

    return {
        "disease": [disease_row],
        "target": targets,
        "disease_target": edges,
        "_total_associated": d["associatedTargets"]["count"],
    }


TARGET_PROFILE = """
query TargetProfile($ensemblId: String!, $nDiseases: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    targetClass { id label }
    proteinIds { id source }
    dbXrefs { id source }
    subcellularLocations { location termSL }
    pathways { pathwayId pathway topLevelTerm }
    tractability { label modality value }
    isEssential
    geneticConstraint { constraintType score exp obs upperBin }
    safetyLiabilities {
      event
      eventId
      datasource
      effects { direction dosing }
      biosamples { tissueLabel cellLabel }
    }
    chemicalProbes { id isHighQuality mechanismOfAction origin }
    mousePhenotypes { modelPhenotypeLabel modelPhenotypeClasses { label } }
    associatedDiseases(page: {index: 0, size: $nDiseases}) {
      count
      rows {
        score
        datatypeScores { id score }
        disease { id name therapeuticAreas { name } }
      }
    }
  }
}
"""


def fetch_target_profile(ensembl_id: str, n_diseases: int = 20) -> dict:
    """Everything Open Targets knows about one target.

    This is the safety half of a dossier. `isEssential` and `geneticConstraint`
    matter more than most people expect: a pan-essential gene whose loss-of-
    function is depleted in the human population is a target where systemic
    inhibition is likely to be toxic, no matter how good the association score.
    """
    data = _query(TARGET_PROFILE, {"ensemblId": ensembl_id, "nDiseases": n_diseases})
    t = data.get("target")
    if t is None:
        raise ValueError(f"Open Targets has no target {ensembl_id!r}")

    loeuf = next(
        (g for g in (t.get("geneticConstraint") or []) if g["constraintType"] == "lof"),
        None,
    )

    return {
        "ensembl_id": t["id"],
        "symbol": t["approvedSymbol"],
        "name": t.get("approvedName"),
        "biotype": t.get("biotype"),
        "function": (t.get("functionDescriptions") or [None])[0],
        "target_class": [c["label"] for c in (t.get("targetClass") or [])],
        "uniprot_ids": [
            p["id"] for p in (t.get("proteinIds") or [])
            if p.get("source") in ("uniprot_swissprot", "uniprot_trembl")
        ],
        "subcellular_locations": sorted(
            {loc["location"] for loc in (t.get("subcellularLocations") or []) if loc.get("location")}
        ),
        "pathways": sorted({p["topLevelTerm"] for p in (t.get("pathways") or []) if p.get("topLevelTerm")}),
        "tractability": [
            {"modality": x["modality"], "label": x["label"]}
            for x in (t.get("tractability") or []) if x.get("value")
        ],
        "is_essential": t.get("isEssential"),
        "loeuf_score": loeuf["score"] if loeuf else None,
        "loeuf_bin": loeuf.get("upperBin") if loeuf else None,
        "safety_liabilities": [
            {
                "event": s.get("event"),
                "datasource": s.get("datasource"),
                "effects": [
                    f"{e.get('direction')}/{e.get('dosing')}" for e in (s.get("effects") or [])
                ],
                "biosamples": sorted(
                    {
                        b.get("tissueLabel") or b.get("cellLabel")
                        for b in (s.get("biosamples") or [])
                        if b.get("tissueLabel") or b.get("cellLabel")
                    }
                )[:5],
            }
            for s in (t.get("safetyLiabilities") or [])
        ],
        "chemical_probes": [
            {
                "id": p["id"],
                "high_quality": p.get("isHighQuality"),
                "moa": p.get("mechanismOfAction") or [],
            }
            for p in (t.get("chemicalProbes") or [])
        ],
        "mouse_phenotypes": sorted(
            {
                m["modelPhenotypeLabel"]
                for m in (t.get("mousePhenotypes") or [])
                if m.get("modelPhenotypeLabel")
            }
        ),
        "n_associated_diseases": t["associatedDiseases"]["count"],
        "associated_diseases": [
            {
                "efo_id": r["disease"]["id"],
                "name": r["disease"]["name"],
                "score": r["score"],
                "therapeutic_areas": [
                    a["name"] for a in (r["disease"].get("therapeuticAreas") or [])
                ],
                "datatype_scores": {
                    s["id"]: s["score"] for s in (r.get("datatypeScores") or [])
                },
            }
            for r in t["associatedDiseases"]["rows"]
        ],
    }
