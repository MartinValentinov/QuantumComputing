"""PubMed via NCBI E-utilities -- the literature layer.

Abstracts and MeSH terms only. Full text is a separate, messier problem
(PMC OA subset, licensing) and is deliberately out of scope for v1: abstracts
plus MeSH indexing already carry most of the signal for triage, and MeSH is
hand-curated, which makes it far more reliable than anything extracted from
free text.

Set NCBI_API_KEY in the environment to raise the rate limit from 3/s to 10/s.
API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/  (key optional)
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

from qc_research.db import now
from qc_research.http import request, request_text

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SOURCE = "pubmed"


def _auth() -> dict[str, str]:
    key = os.environ.get("NCBI_API_KEY")
    return {"api_key": key} if key else {}


def search(query: str, retmax: int = 100, mindate: int | None = None) -> list[str]:
    """Run a PubMed query, return PMIDs.

    `query` takes full PubMed syntax, so field tags work:
        'KRAS[Title/Abstract] AND pancreatic neoplasms[MeSH Terms]'
    Use them. An untagged query matches far too much.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        **_auth(),
    }
    if mindate:
        params |= {"mindate": mindate, "maxdate": 3000, "datetype": "pdat"}

    payload = request(f"{BASE}/esearch.fcgi", params=params)
    return payload.get("esearchresult", {}).get("idlist", [])


def fetch_articles(pmids: list[str], batch: int = 100) -> list[dict]:
    """Fetch metadata + abstracts for a list of PMIDs."""
    ts = now()
    rows: list[dict] = []

    for i in range(0, len(pmids), batch):
        chunk = pmids[i : i + batch]
        xml = request_text(
            f"{BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(chunk),
                "retmode": "xml",
                **_auth(),
            },
        )
        root = ET.fromstring(xml)

        for art in root.findall(".//PubmedArticle"):
            cit = art.find(".//MedlineCitation")
            if cit is None:
                continue

            pmid = _text(cit.find("PMID"))
            article = cit.find("Article")
            if article is None or pmid is None:
                continue

            parts = []
            for ab in article.findall(".//Abstract/AbstractText"):
                label = ab.get("Label")
                body = "".join(ab.itertext()).strip()
                parts.append(f"{label}: {body}" if label else body)

            year = _text(article.find(".//Journal/JournalIssue/PubDate/Year"))
            if not year:
                medline = _text(article.find(".//Journal/JournalIssue/PubDate/MedlineDate"))
                year = medline[:4] if medline else None

            authors = []
            for a in article.findall(".//AuthorList/Author"):
                last, initials = _text(a.find("LastName")), _text(a.find("Initials"))
                if last:
                    authors.append(f"{last} {initials}" if initials else last)

            doi = None
            for aid in art.findall(".//ArticleIdList/ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = _text(aid)

            rows.append(
                {
                    "pmid": pmid,
                    "doi": doi,
                    "title": _text(article.find("ArticleTitle")),
                    "abstract": "\n\n".join(parts) or None,
                    "journal": _text(article.find(".//Journal/ISOAbbreviation")),
                    "year": int(year) if year and year.isdigit() else None,
                    "authors": json.dumps(authors),
                    "mesh_terms": json.dumps(
                        [
                            _text(mh.find("DescriptorName"))
                            for mh in cit.findall(".//MeshHeadingList/MeshHeading")
                        ]
                    ),
                    "pub_types": json.dumps(
                        [_text(pt) for pt in article.findall(".//PublicationTypeList/PublicationType")]
                    ),
                    "source": SOURCE,
                    "fetched_at": ts,
                }
            )
    return rows


def target_disease_query(symbol: str, disease: str, review_only: bool = False) -> str:
    """Build a properly field-tagged query for a target/disease pair."""
    q = (
        f'("{symbol}"[Title/Abstract] OR "{symbol}"[Gene Name]) '
        f'AND "{disease}"[Title/Abstract]'
    )
    if review_only:
        q += ' AND (review[Publication Type] OR systematic review[Publication Type])'
    return q


def _text(node) -> str | None:
    if node is None:
        return None
    return ("".join(node.itertext()) or "").strip() or None
