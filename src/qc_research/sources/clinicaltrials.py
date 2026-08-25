"""ClinicalTrials.gov v2 -- what was actually tried in humans, and what happened.

This is the most underused source in computational drug discovery. Terminated
and withdrawn trials, plus the free-text `whyStopped` field, are the closest
thing to a public record of failure that the field has. Modelling only
successes teaches you nothing about risk.

API: https://clinicaltrials.gov/api/v2/  (no key needed)
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from qc_research.db import now
from qc_research.http import request

BASE = "https://clinicaltrials.gov/api/v2"
SOURCE = "clinicaltrials.gov"


def _studies(params: dict[str, Any], limit: int) -> Iterator[dict]:
    """Walk the token-based pagination."""
    fetched = 0
    token = None
    while fetched < limit:
        page_params = {**params, "pageSize": min(100, limit - fetched), "format": "json"}
        if token:
            page_params["pageToken"] = token
        page = request(f"{BASE}/studies", params=page_params)
        studies = page.get("studies", [])
        if not studies:
            return
        for s in studies:
            yield s
            fetched += 1
            if fetched >= limit:
                return
        token = page.get("nextPageToken")
        if not token:
            return


def fetch_trials(
    condition: str,
    limit: int = 200,
    intervention: str | None = None,
    include_all_statuses: bool = True,
) -> list[dict]:
    """Trials for a condition, including the failed ones.

    `include_all_statuses` defaults to True on purpose -- the default view on
    the website hides terminated studies, which are the informative ones.
    """
    params: dict[str, Any] = {"query.cond": condition}
    if intervention:
        params["query.intr"] = intervention
    if not include_all_statuses:
        params["filter.overallStatus"] = "COMPLETED|ACTIVE_NOT_RECRUITING|RECRUITING"

    ts = now()
    rows = []
    for s in _studies(params, limit):
        p = s.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        status = p.get("statusModule", {})
        design = p.get("designModule", {})
        arms = p.get("armsInterventionsModule", {})
        outcomes = p.get("outcomesModule", {})
        cond_mod = p.get("conditionsModule", {})
        sponsor_mod = p.get("sponsorCollaboratorsModule", {})

        enrollment = (design.get("enrollmentInfo") or {}).get("count")
        phases = design.get("phases") or []

        rows.append(
            {
                "nct_id": ident.get("nctId"),
                "title": ident.get("briefTitle"),
                "status": status.get("overallStatus"),
                "phase": "/".join(phases) if phases else None,
                "enrollment": int(enrollment) if enrollment is not None else None,
                "start_date": (status.get("startDateStruct") or {}).get("date"),
                "completion_date": (
                    status.get("completionDateStruct") or {}
                ).get("date"),
                "why_stopped": status.get("whyStopped"),
                "sponsor": (
                    (sponsor_mod.get("leadSponsor") or {}).get("name")
                ),
                "conditions": json.dumps(cond_mod.get("conditions") or []),
                "interventions": json.dumps(
                    [
                        {"type": i.get("type"), "name": i.get("name")}
                        for i in (arms.get("interventions") or [])
                    ]
                ),
                "primary_outcomes": json.dumps(
                    [
                        {
                            "measure": o.get("measure"),
                            "timeFrame": o.get("timeFrame"),
                        }
                        for o in (outcomes.get("primaryOutcomes") or [])
                    ]
                ),
                "has_results": int(bool(s.get("hasResults"))),
                "source": SOURCE,
                "fetched_at": ts,
            }
        )
    return rows


def failure_analysis(trials: list[dict]) -> dict:
    """Summarise the attrition pattern. The point of pulling trials at all."""
    from collections import Counter

    by_status = Counter(t["status"] for t in trials)
    by_phase = Counter(t["phase"] or "N/A" for t in trials)
    stopped = [
        {"nct_id": t["nct_id"], "phase": t["phase"], "reason": t["why_stopped"]}
        for t in trials
        if t.get("why_stopped")
    ]
    return {
        "n_trials": len(trials),
        "by_status": dict(by_status.most_common()),
        "by_phase": dict(by_phase.most_common()),
        "n_with_stated_failure_reason": len(stopped),
        "stopped": stopped,
    }
