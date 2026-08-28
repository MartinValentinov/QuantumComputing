"""Tiny HTTP layer: stdlib only, polite, cached.

Every public biomedical API here is free and unauthenticated, but they all
rate-limit. Caching to disk means a re-run costs nothing and you can iterate
on parsing without hammering anyone.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.environ.get("QC_CACHE_DIR", "data/cache"))
USER_AGENT = "qc-research/0.1 (research use; contact via github.com/MartinValentinov/QC)"

_MIN_INTERVAL = {
    "eutils.ncbi.nlm.nih.gov": 0.35,
    "www.ebi.ac.uk": 0.2,
    "api.platform.opentargets.org": 0.2,
    "clinicaltrials.gov": 0.2,
}
_last_call: dict[str, float] = {}


class HttpError(RuntimeError):
    pass


def _throttle(host: str) -> None:
    interval = _MIN_INTERVAL.get(host, 0.1)
    elapsed = time.monotonic() - _last_call.get(host, 0.0)
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_call[host] = time.monotonic()


def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return CACHE_DIR / digest[:2] / f"{digest}.json"


def request(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict | None = None,
    headers: dict[str, str] | None = None,
    use_cache: bool = True,
    timeout: int = 60,
    retries: int = 3,
) -> Any:
    """GET (or POST when `json_body` is given) and parse JSON.

    Raises HttpError after exhausting retries. 4xx other than 429 is not
    retried -- it means the request itself is wrong.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"

    cache_key = url + (json.dumps(json_body, sort_keys=True) if json_body else "")
    cache_file = _cache_path(cache_key)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    host = urllib.parse.urlparse(url).netloc
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode()
        req_headers["Content-Type"] = "application/json"

    last_err: Exception | None = None
    for attempt in range(retries):
        _throttle(host)
        req = urllib.request.Request(url, data=data, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw) if raw.strip() else {}
            if use_cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload))
            return payload
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise HttpError(f"{e.code} {e.reason} for {url}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise HttpError(f"failed after {retries} attempts: {url} ({last_err})")


def request_text(url: str, *, params: dict | None = None, timeout: int = 60) -> str:
    """Same, for endpoints that return XML rather than JSON (PubMed efetch)."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"

    cache_file = _cache_path(url + "::text")
    if cache_file.exists():
        return cache_file.read_text()

    _throttle(urllib.parse.urlparse(url).netloc)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text)
    return text
