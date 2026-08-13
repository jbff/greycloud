"""
Discovery Engine search-and-ground utilities for GreyCloud.

Explicit retrieval for grounded generation: instead of relying on the
server-side ``tools.retrieval.vertexAiSearch`` mechanism (broken for the
Gemini 3.x line, see GREYCLOUD_GROUNDING_DIAGNOSIS.md), query the Vertex AI
Search / Discovery Engine ``:search`` API directly with GreyCloud's existing
credentials and shape the results into citation-able grounding sources.

This module is deliberately self-contained: it never raises on search failure
(``search_sources`` / ``asearch_sources`` log a warning and return ``[]``) so
the generation path can degrade to *generation without context* instead of
failing the request.
"""

import asyncio
import html as _html
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import httpx
import requests

from .auth import get_credentials
from .config import GreyCloudConfig

logger = logging.getLogger(__name__)

# Discovery Engine endpoints keyed by the `locations/<loc>` segment of the
# datastore resource path. Anything else falls back to the global endpoint.
_ENDPOINTS = {
    "global": "https://discoveryengine.googleapis.com",
    "us": "https://us-discoveryengine.googleapis.com",
    "eu": "https://eu-discoveryengine.googleapis.com",
}

_GLOBAL_ENDPOINT = _ENDPOINTS["global"]

# Retry policy: up to _MAX_ATTEMPTS total attempts (i.e. 2 retries) with a
# short exponential backoff on transport errors, HTTP 429 (Discovery Engine
# throttling), and 5xx responses.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.5

_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")

# Instruction line appended after the grounding block, mirroring the app's
# knowledge-block pattern (diagnosis 4.3.2): quote only from these sources and
# cite each quote with its [n] citation number.
_INSTRUCTION = (
    "Quote only from the sources above, and cite each quoted passage with its "
    "[n] citation number in brackets."
)


@dataclass
class GroundingSource:
    """A single retrieved source ready to be injected into a prompt."""

    title: str
    link: str  # source URI, e.g. gs://bucket/file.pdf
    snippet: str  # cleaned passage text (HTML tags stripped)
    index: int  # 1-based citation number [1]..[n]


def discovery_endpoint_for_datastore(datastore_path: str) -> str:
    """Select the Discovery Engine endpoint matching the datastore's region.

    Parses the ``locations/<loc>`` segment out of the resource path
    (``projects/P/locations/<loc>/collections/default_collection/dataStores/D``):

    - ``global`` -> ``https://discoveryengine.googleapis.com``
    - ``us``     -> ``https://us-discoveryengine.googleapis.com``
    - ``eu``     -> ``https://eu-discoveryengine.googleapis.com``

    Any other (or missing) location logs a warning and returns the global
    endpoint.
    """
    if not datastore_path:
        logger.warning(
            "discovery_endpoint_for_datastore: empty datastore path; "
            "using global endpoint"
        )
        return _GLOBAL_ENDPOINT

    match = re.search(r"locations/([^/]+)/", datastore_path)
    if match:
        loc = match.group(1)
        endpoint = _ENDPOINTS.get(loc)
        if endpoint is not None:
            return endpoint
        logger.warning(
            "discovery_endpoint_for_datastore: unknown datastore location %r; "
            "falling back to global Discovery Engine endpoint",
            loc,
        )
        return _GLOBAL_ENDPOINT

    logger.warning(
        "discovery_endpoint_for_datastore: could not parse a location from "
        "datastore path %r; using global Discovery Engine endpoint",
        datastore_path,
    )
    return _GLOBAL_ENDPOINT


def _clean_snippet(text: Optional[str]) -> str:
    """Strip HTML tags, decode common entities, and collapse whitespace.

    ``html.unescape`` runs first so entity-encoded tags (e.g. ``&lt;b&gt;``)
    are decoded before tag stripping; ``&nbsp;`` (-> ``\\xa0``) is normalized
    to a regular space.
    """
    if not text:
        return ""
    text = _html.unescape(str(text))
    text = _TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _shape_results(data: dict, max_chars: int) -> List[GroundingSource]:
    """Extract GroundingSources from a Discovery Engine ``:search`` response.

    Each ``results[i].document.derivedStructData`` contributes ``title``,
    ``link``, and the first ``snippets[*].snippet`` (HTML-cleaned). ``index``
    is 1-based by relevance order. The total snippet text kept is bounded by
    ``max_chars`` (snippets truncated from the end), mirroring the token budget
    cap used when injecting the context block.
    """
    if not isinstance(data, dict):
        return []

    sources: List[GroundingSource] = []
    results = data.get("results") or []
    remaining = max(0, max_chars)

    for i, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        doc = item.get("document") or {}
        if not isinstance(doc, dict):
            doc = {}
        dsd = doc.get("derivedStructData") or {}
        if not isinstance(dsd, dict):
            dsd = {}

        title = dsd.get("title") or ""
        link = dsd.get("link") or ""

        raw_snippet = ""
        # Malformed-shape responses may carry a non-list ``snippets`` (e.g. a
        # dict); treat that as "no snippet" instead of raising inside the
        # enclosing try (which would burn retry attempts on a shape we can't fix).
        snippets = dsd.get("snippets") or []
        if isinstance(snippets, list) and snippets:
            first = snippets[0]
            if isinstance(first, dict):
                raw_snippet = first.get("snippet", "")
            else:
                raw_snippet = str(first)
        snippet = _clean_snippet(raw_snippet)

        # Bound total snippet text: truncate later snippets first.
        if snippet:
            if remaining <= 0:
                snippet = ""
            elif len(snippet) > remaining:
                snippet = snippet[:remaining].rstrip()
            remaining -= len(snippet)

        sources.append(
            GroundingSource(title=title, link=link, snippet=snippet, index=i + 1)
        )

    return sources


def _build_headers(config: GreyCloudConfig) -> Tuple[Optional[dict], Optional[str]]:
    """Resolve auth headers for a Discovery Engine search call.

    OAuth path: ``Authorization: Bearer <token>`` (via ``get_credentials`` +
    ``google.auth.transport.requests.Request()``) plus
    ``x-goog-user-project: {project_id}``. API-key path:
    ``x-goog-api-key: <key>``.

    Returns ``(headers, None)`` on success or ``(None, error_message)`` on
    failure; callers must treat the latter as a search failure and return ``[]``.
    """
    try:
        creds = get_credentials(
            project_id=getattr(config, "project_id", ""),
            sa_email=getattr(config, "sa_email", None),
            use_api_key=bool(getattr(config, "use_api_key", False)),
            api_key_file=getattr(config, "api_key_file", "GOOGLE_CLOUD_API_KEY"),
            auto_reauth=bool(getattr(config, "auto_reauth", True)),
        )
    except Exception as e:  # noqa: BLE001 - must never raise to the caller
        return None, f"failed to resolve credentials: {e}"

    if getattr(config, "use_api_key", False):
        return {"x-goog-api-key": creds}, None

    try:
        # Refresh only when needed: google.auth.default() returns a module-cached
        # credentials object, and each refresh of impersonated credentials mints
        # a fresh token (one IAM call), so refreshing a still-valid token on
        # every search wastes a metadata-server hit / refresh-token grant.
        # _StaticTokenCredentials sets ``token`` and ``expired=False``, so this
        # guard is a no-op there.
        from google.auth.transport.requests import Request

        if creds.token is None or getattr(creds, "expired", True):
            creds.refresh(Request())
        token = creds.token
    except Exception as e:  # noqa: BLE001 - must never raise to the caller
        return None, f"failed to obtain access token: {e}"

    headers = {
        "Authorization": f"Bearer {token}",
        "x-goog-user-project": getattr(config, "project_id", ""),
    }
    return headers, None


def _search_url(datastore_path: str) -> str:
    endpoint = discovery_endpoint_for_datastore(datastore_path)
    return f"{endpoint}/v1/{datastore_path}/servingConfigs/default_search:search"


def _search_payload(query: str, page_size: int) -> dict:
    return {
        "query": query,
        "pageSize": max(1, page_size),
        "contentSearchSpec": {"snippetSpec": {"returnSnippet": True}},
    }


def _validate_args(
    config: GreyCloudConfig, query: str, page_size: int
) -> Optional[str]:
    """Return an error message when the search cannot run, else None.

    Also rejects wrong-typed arguments (non-string datastore/query, non-int
    page_size) so no argument-value class can raise out of the search
    functions and break the never-raise contract.
    """
    if not query or not str(query).strip():
        return "empty query; skipping Discovery Engine search"
    if not isinstance(query, str):
        return "query must be a string; skipping Discovery Engine search"
    datastore = getattr(config, "vertex_ai_search_datastore", None)
    if not datastore:
        return (
            "GreyCloudConfig.vertex_ai_search_datastore is not set; "
            "skipping Discovery Engine search"
        )
    if not isinstance(datastore, str):
        return (
            "GreyCloudConfig.vertex_ai_search_datastore must be a string; "
            "skipping Discovery Engine search"
        )
    if not isinstance(page_size, int):
        return "page_size must be an integer; skipping Discovery Engine search"
    return None


def search_sources(
    config: GreyCloudConfig,
    query: str,
    page_size: int = 5,
    max_chars: int = 8000,
    timeout: float = 30.0,
) -> List[GroundingSource]:
    """Search the configured Discovery Engine datastore (sync).

    Never raises: on any exception, non-2xx response, or invalid args it logs a
    warning and returns ``[]`` so the caller can generate without context.

    Args:
        config: GreyCloudConfig with ``vertex_ai_search_datastore`` set.
        query: User query to search for.
        page_size: Number of results to request (maps to ``pageSize``).
        max_chars: Budget for total snippet text kept across sources.
        timeout: Per-request timeout in seconds.

    Returns:
        List of up to ``page_size`` GroundingSources, ordered by relevance.
    """
    invalid = _validate_args(config, query, page_size)
    if invalid:
        logger.warning("search_sources: %s", invalid)
        return []

    headers, auth_error = _build_headers(config)
    if auth_error:
        logger.warning("search_sources: %s", auth_error)
        return []

    datastore = getattr(config, "vertex_ai_search_datastore")
    last_exc: Optional[BaseException] = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            url = _search_url(datastore)
            payload = _search_payload(query, page_size)
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code >= 500 or response.status_code == 429:
                last_exc = RuntimeError(
                    f"Discovery Engine search returned HTTP {response.status_code}"
                )
                logger.warning(
                    "search_sources: attempt %d failed with HTTP %d; retrying",
                    attempt + 1,
                    response.status_code,
                )
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                logger.warning(
                    "search_sources: Discovery Engine search failed with HTTP %s: %s",
                    response.status_code,
                    response.text[:300],
                )
                return []
            try:
                data = response.json()
            except ValueError as e:
                # Server-data problem (unparseable body), not transient: do not retry.
                logger.warning(
                    "search_sources: invalid JSON in Discovery Engine response "
                    "(HTTP %s): %s",
                    response.status_code,
                    e,
                )
                return []
            return _shape_results(data, max_chars)
        except Exception as e:  # noqa: BLE001 - must never raise to the caller
            last_exc = e
            logger.warning(
                "search_sources: attempt %d failed: %s; retrying",
                attempt + 1,
                e,
            )
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))

    logger.warning(
        "search_sources: Discovery Engine search failed after %d attempts: %s",
        _MAX_ATTEMPTS,
        last_exc,
    )
    return []


async def asearch_sources(
    config: GreyCloudConfig,
    query: str,
    page_size: int = 5,
    max_chars: int = 8000,
    timeout: float = 30.0,
) -> List[GroundingSource]:
    """Search the configured Discovery Engine datastore (async).

    Same contract as :func:`search_sources`: never raises, returns ``[]`` on
    any failure so the caller can generate without context.
    """
    invalid = _validate_args(config, query, page_size)
    if invalid:
        logger.warning("asearch_sources: %s", invalid)
        return []

    headers, auth_error = _build_headers(config)
    if auth_error:
        logger.warning("asearch_sources: %s", auth_error)
        return []

    datastore = getattr(config, "vertex_ai_search_datastore")
    last_exc: Optional[BaseException] = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            url = _search_url(datastore)
            payload = _search_payload(query, page_size)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 500 or response.status_code == 429:
                last_exc = RuntimeError(
                    f"Discovery Engine search returned HTTP {response.status_code}"
                )
                logger.warning(
                    "asearch_sources: attempt %d failed with HTTP %d; retrying",
                    attempt + 1,
                    response.status_code,
                )
                if attempt < _MAX_ATTEMPTS - 1:
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                logger.warning(
                    "asearch_sources: Discovery Engine search failed with HTTP %s: %s",
                    response.status_code,
                    response.text[:300],
                )
                return []
            try:
                data = response.json()
            except ValueError as e:
                # Server-data problem (unparseable body), not transient: do not retry.
                logger.warning(
                    "asearch_sources: invalid JSON in Discovery Engine response "
                    "(HTTP %s): %s",
                    response.status_code,
                    e,
                )
                return []
            return _shape_results(data, max_chars)
        except Exception as e:  # noqa: BLE001 - must never raise to the caller
            last_exc = e
            logger.warning(
                "asearch_sources: attempt %d failed: %s; retrying",
                attempt + 1,
                e,
            )
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))

    logger.warning(
        "asearch_sources: Discovery Engine search failed after %d attempts: %s",
        _MAX_ATTEMPTS,
        last_exc,
    )
    return []


def build_grounding_context(
    sources: Sequence[GroundingSource], max_chars: int = 8000
) -> str:
    """Render GroundingSources into an injectable context block (pure function).

    Format::

        <grounding_sources>
        [1] (Title — gs://source-uri.pdf)
        "Snippet passage text ..."
        [2] (Title2 — gs://source2.pdf)
        "Snippet2 ..."
        </grounding_sources>
        Quote only from the sources above, and cite each quoted passage with
        its [n] citation number in brackets.

    The block never exceeds ``max_chars`` characters: snippets are truncated
    from the end (later sources first) until the block fits. Returns the empty
    string for no sources.
    """
    sources = list(sources)
    if not sources:
        return ""

    citation_lines = [f"[{s.index}] ({s.title} — {s.link})" for s in sources]
    n = len(sources)

    # Exact length of the rendered block with empty snippets:
    # header + 2 lines per source (citation + quoted snippet) + footer + instruction,
    # with 2 quote chars and 2 newlines per source plus 2 newlines for the
    # header/footer/instruction boundaries.
    fixed_len = (
        len("<grounding_sources>")
        + sum(len(c) for c in citation_lines)
        + len("</grounding_sources>")
        + len(_INSTRUCTION)
        + 4 * n
        + 2
    )

    if fixed_len >= max_chars:
        # Metadata alone doesn't fit; hard-truncate to honor the cap.
        block = _render_block(citation_lines, [""] * n)
        return block[: max(0, max_chars)]

    budget = max_chars - fixed_len
    snippet_texts: List[str] = []
    used = 0
    for s in sources:
        text = s.snippet or ""
        remaining = budget - used
        if remaining <= 0:
            snippet_texts.append("")
            continue
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        snippet_texts.append(text)
        used += len(text)

    return _render_block(citation_lines, snippet_texts)


def _render_block(citation_lines: Sequence[str], snippet_texts: Sequence[str]) -> str:
    lines = ["<grounding_sources>"]
    for citation, snippet in zip(citation_lines, snippet_texts):
        lines.append(citation)
        lines.append(f'"{snippet}"')
    lines.append("</grounding_sources>")
    lines.append(_INSTRUCTION)
    return "\n".join(lines)
