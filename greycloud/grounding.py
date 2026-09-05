"""
Discovery Engine search-and-ground utilities for GreyCloud.

Explicit retrieval for grounded generation: instead of relying on the
server-side ``tools.retrieval.vertexAiSearch`` mechanism (broken for the
Gemini 3.x line), query the Vertex AI
Search / Discovery Engine ``:search`` API directly with GreyCloud's existing
credentials and shape the results into citation-able grounding sources.

This module is deliberately self-contained: it never raises on search failure
(``search_sources`` / ``asearch_sources`` log the failure and return ``[]`` —
WARNING for ordinary failures, ERROR for the chunking-config 400 when
``extractive_content_spec`` is on) so the generation path can degrade to
*generation without context* instead of failing the request.
"""

import asyncio
import html as _html
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

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

# Floor for the per-source fair share of the character budget (see
# _shape_results): a long result list must not shrink an individual source's
# share to a useless fragment.
_MIN_PER_SOURCE_CHARS = 250

# Quote characters Discovery Engine treats as exact-phrase delimiters: the
# ASCII straight quote plus the curly (U+201C/U+201D) and fullwidth (U+FF02)
# forms users paste from word processors / CJK input. Single quotes and
# apostrophes are deliberately excluded (see _normalize_query docstring).
_QUOTE_RE = re.compile(r'["“”＂]')

# Only *balanced* quote pairs are phrase delimiters. A lone quote is a unit
# mark (inches/seconds/ditto, e.g. '15" laptop') and must survive; stripping
# it would silently destroy the unit (review finding #6). The pair must
# enclose content that starts and ends with a word character, so two unit
# marks ('5" 6"') and quotes-only queries ('"""') are left untouched.
_QUOTE_PAIR_RE = re.compile(r'["“”＂](?=[^"“”＂\s])([^"“”＂]+?)(?<=[^"“”＂\s])["“”＂]')

# Matches the chunking-config 400's field mention in the spellings it can
# plausibly appear in: the snake_case form Google's message uses
# (max_extractive_answer_count), the camelCase wire name
# (maxExtractiveAnswerCount), and hyphen/space variants.
_CHUNKING_MARKER_RE = re.compile(
    r"max[\s_-]*extractive[\s_-]*answer[\s_-]*count", re.IGNORECASE
)


def _collapse_ws(text: str) -> str:
    """Collapse runs of whitespace (incl. ``\\xa0``) to single spaces and trim."""
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip()


def _has_searchable_content(query: str) -> bool:
    """True if the query has any content beyond quote characters.

    A quotes-only query (e.g. ``'\"\"\"'``) has no searchable tokens once the
    phrase delimiters are removed; sending it to Discovery Engine is a wasted
    call (review finding #1).
    """
    return bool(_collapse_ws(_QUOTE_RE.sub(" ", query)))


# Instruction line appended after the grounding block, mirroring the app's
# knowledge-block pattern (diagnosis 4.3.2): when the model quotes, it quotes
# only from these sources and cites each quote with its [n] citation number.
# Deliberately conditional (RAG proposal item 3): the unconditional phrasing
# implied every response must contain a quote/citation, and because the block
# is prepended to the last user message the instruction outranked the system
# prompt — callers saw citations leak into pleasantries and sign-offs.
_INSTRUCTION = (
    "When you quote from the sources above, quote only from them, and cite "
    "each quoted passage with its [n] citation number in brackets."
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
    return _collapse_ws(text)


def _shape_results(data: dict, max_chars: int) -> List[GroundingSource]:
    """Extract GroundingSources from a Discovery Engine ``:search`` response.

    Each ``results[i].document.derivedStructData`` contributes ``title``,
    ``link``, and a passage (HTML-cleaned). Passages prefer the first
    ``extractiveAnswers[*].content`` (paragraph-scale, quote-ready text; the
    REST API returns the camelCase key, but the snake_case spelling is also
    accepted) and fall back to the first ``snippets[*].snippet`` when no
    non-empty answer exists. ``index`` is 1-based by relevance order.

    The total passage text kept is bounded by ``max_chars`` (truncated from
    the end), mirroring the token budget cap used when injecting the context
    block. Because extractive answers are paragraph-scale, each source is
    additionally capped at an even share of the budget (with a floor so a
    long result list cannot shrink the share to nothing) — otherwise the
    first answers eat the whole budget and later sources keep nothing.
    """
    if not isinstance(data, dict):
        # Malformed body (not a JSON object): a server-data problem like
        # invalid JSON, not transient — raise so the caller's non-retryable
        # ValueError path handles it instead of burning retry attempts on a
        # shape we can't fix.
        raise ValueError(
            f"Discovery Engine response body is not a JSON object: "
            f"{type(data).__name__}"
        )

    sources: List[GroundingSource] = []
    results = data.get("results") or []
    if not isinstance(results, list):
        # Malformed shape (e.g. a truthy non-sized value such as an int):
        # same reasoning as above — raise into the caller's non-retryable
        # ValueError path rather than burning retries on a shape we can't fix.
        raise ValueError(
            f"Discovery Engine response 'results' field is not a list: "
            f"{type(results).__name__}"
        )
    remaining = max(0, max_chars)
    per_source_cap = (
        max(_MIN_PER_SOURCE_CHARS, max_chars // len(results)) if results else 0
    )

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

        raw_passage = ""
        # Extractive answers first (paragraph-scale); the REST API uses the
        # camelCase key. Malformed shapes (non-list) are treated as "no
        # answers" rather than raising inside the enclosing try, which would
        # burn retry attempts on a shape we can't fix.
        for key in ("extractiveAnswers", "extractive_answers"):
            answers = dsd.get(key) or []
            if isinstance(answers, list):
                for answer in answers:
                    if isinstance(answer, dict):
                        content = _clean_snippet(answer.get("content"))
                        if content:
                            raw_passage = content
                            break
            if raw_passage:
                break
        # Snippet fallback (1-3 sentence keyword-context fragment).
        if not raw_passage:
            # Malformed-shape responses may carry a non-list ``snippets`` (e.g. a
            # dict); treat that as "no snippet" instead of raising inside the
            # enclosing try (which would burn retry attempts on a shape we can't fix).
            snippets = dsd.get("snippets") or []
            if isinstance(snippets, list) and snippets:
                first = snippets[0]
                if isinstance(first, dict):
                    raw_passage = first.get("snippet", "")
                else:
                    raw_passage = str(first)
        snippet = _clean_snippet(raw_passage)

        # Fair share of the budget, then bound the total: truncate later
        # sources first once the budget runs out.
        if snippet and per_source_cap and len(snippet) > per_source_cap:
            snippet = snippet[:per_source_cap].rstrip()
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


def _search_payload(
    query: str, page_size: int, extractive_content_spec: bool = False
) -> dict:
    """Build the Discovery Engine ``:search`` request body.

    The snippets-only ``contentSearchSpec`` is the default: it is accepted by
    every datastore type. ``extractive_content_spec=True`` additionally
    requests paragraph-scale extractive answers — quote-ready text rather than
    1-3 sentence keyword fragments — but datastores created with chunking
    config reject the field with HTTP 400, so which content types the
    datastore supports is the caller's knowledge to opt into, never a
    library-side default (0.3.12–0.3.14 sent it unconditionally and broke
    every chunked datastore).
    """
    content_search_spec: dict = {"snippetSpec": {"returnSnippet": True}}
    if extractive_content_spec:
        content_search_spec["extractiveContentSpec"] = {
            "maxExtractiveAnswerCount": 2,
        }
    return {
        "query": query,
        "pageSize": max(1, page_size),
        "contentSearchSpec": content_search_spec,
    }


def _log_search_http_error(prefix: str, response, extractive_requested: bool) -> None:
    """Log a non-2xx ``:search`` response at the appropriate level.

    A 400 whose body blames the extractive-answer limit while the caller
    asked for the extractive spec means the datastore uses chunking config:
    log at ERROR with the disable hint (never silently retry into a
    different mode — the caller asked for extractive, so the failure must be
    loud, not downgraded). Every other non-2xx logs the standard WARNING.

    The marker is matched against the *full* body (the mention can sit past
    any truncation point, e.g. inside ``error.details``) and accepts the
    snake_case spelling Google's 400 uses as well as the camelCase wire name
    and hyphen/space variants, so a vendor rewording that keeps the field
    name still trips the loud path. The body is truncated only for the log
    line itself.
    """
    body = response.text
    if (
        response.status_code == 400
        and extractive_requested
        and _CHUNKING_MARKER_RE.search(body)
    ):
        logger.error(
            "%s: Discovery Engine search failed with HTTP 400: %s — the datastore "
            "likely uses 'chunking config', which rejects extractiveContentSpec; "
            "set GreyCloudConfig.extractive_content_spec=False (the default) to "
            "search this datastore",
            prefix,
            body[:300],
        )
    else:
        logger.warning(
            "%s: Discovery Engine search failed with HTTP %s: %s",
            prefix,
            response.status_code,
            body[:300],
        )


def _normalize_query(query: str) -> str:
    """Make a user query safe for Discovery Engine keyword search.

    Discovery Engine treats double-quoted phrases as exact contiguous verbatim
    matches. A quoted long title exists in document *metadata*, not as
    contiguous body text, so the exact-phrase requirement can never be
    satisfied and keyword AND-semantics collapse the whole query to 0 results
    (diagnosis section 1.5). Replace each *balanced* quote pair with a space so
    every token participates as an ordinary keyword without adjacent tokens
    merging (``"renewable"energy`` -> ``renewable energy``), and trim / collapse
    internal whitespace.

    Covers the ASCII straight quote plus the curly (U+201C/U+201D) and
    fullwidth (U+FF02) forms users paste from word processors / CJK input.
    Single quotes and apostrophes are left untouched, and a lone quote is
    preserved as a unit mark (``'15" laptop'`` keeps its inch mark).

    Never raises: non-string input is returned unchanged, so the
    never-raise/degrade-to-ungrounded contract of the search functions is
    preserved.
    """
    if not isinstance(query, str):
        return query
    stripped = _QUOTE_PAIR_RE.sub(r" \1 ", query)
    normalized = _collapse_ws(stripped)
    result = normalized or query
    if result != query and logger.isEnabledFor(logging.DEBUG):
        n = len(_QUOTE_RE.findall(query)) - len(_QUOTE_RE.findall(result))
        if n:
            logger.debug(
                "stripped %d quote character(s) from Discovery Engine search query",
                n,
            )
        else:
            logger.debug("collapsed whitespace in Discovery Engine search query")
    return result


def _validate_args(
    config: GreyCloudConfig, query: str, page_size: int
) -> Optional[str]:
    """Return an error message when the search cannot run, else None.

    Also rejects wrong-typed arguments (non-string datastore/query, non-int
    page_size) so no argument-value class can raise out of the search
    functions and break the never-raise contract.
    """
    # Type checks come first: a wrong-typed value (e.g. an object whose
    # __str__/__bool__ raises) must be rejected by isinstance alone, never
    # coerced — coercion could raise out of the search functions and break
    # the never-raise contract.
    if not isinstance(query, str):
        return "query must be a string; skipping Discovery Engine search"
    if not query.strip():
        return "empty query; skipping Discovery Engine search"
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


def _search_sources_once(
    config: GreyCloudConfig,
    query: str,
    page_size: int,
    max_chars: int,
    timeout: float,
) -> Tuple[List[GroundingSource], bool]:
    """Run the Discovery Engine search once, with the retry policy.

    Returns ``(sources, zero_results)``: ``sources`` is the shaped result list
    (``[]`` on any failure), and ``zero_results`` is True only when the search
    succeeded (HTTP 200) but returned no results — the signal the caller uses
    to decide whether to retry with the unquoted query.
    """
    headers, auth_error = _build_headers(config)
    if auth_error:
        logger.warning("search_sources: %s", auth_error)
        return [], False

    datastore = getattr(config, "vertex_ai_search_datastore")
    # Read the flag once: the payload and the 400 log classification must
    # agree, and a divergent second read could misclassify the failure.
    spec = bool(getattr(config, "extractive_content_spec", False))
    last_exc: Optional[BaseException] = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            url = _search_url(datastore)
            payload = _search_payload(query, page_size, extractive_content_spec=spec)
            response = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )
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
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                _log_search_http_error(
                    "search_sources",
                    response,
                    extractive_requested=spec,
                )
                return [], False
            try:
                data = response.json()
                sources = _shape_results(data, max_chars)
            except ValueError as e:
                # Server-data problem (unparseable body, or a body whose shape
                # we can't interpret), not transient: do not retry.
                logger.warning(
                    "search_sources: invalid or malformed Discovery Engine "
                    "response (HTTP %s): %s",
                    response.status_code,
                    e,
                )
                return [], False
            return sources, not sources
        except Exception as e:  # noqa: BLE001 - must never raise to the caller
            last_exc = e
            logger.warning(
                "search_sources: attempt %d failed: %s; retrying",
                attempt + 1,
                e,
            )
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

    logger.warning(
        "search_sources: Discovery Engine search failed after %d attempts: %s",
        _MAX_ATTEMPTS,
        last_exc,
    )
    return [], False


def search_sources(
    config: GreyCloudConfig,
    query: str,
    page_size: int = 5,
    max_chars: int = 8000,
    timeout: float = 30.0,
    retry_unquoted: bool = True,
) -> List[GroundingSource]:
    """Search the configured Discovery Engine datastore (sync).

    Never raises: on any exception, non-2xx response, or invalid args it logs
    the failure and returns ``[]`` so the caller can generate without context.
    Ordinary failures log at WARNING; the chunking-config 400 (when
    ``extractive_content_spec`` is on) logs at ERROR with a disable hint.

    Discovery Engine treats double-quoted phrases as exact contiguous verbatim
    matches, which can collapse a keyword query to 0 results (diagnosis 1.5).
    When the query contains quote characters, the quoted form is searched
    first to preserve exact-phrase semantics; if it returns 0 results, the
    search is retried with the unquoted form (see ``retry_unquoted``).

    Args:
        config: GreyCloudConfig with ``vertex_ai_search_datastore`` set.
        query: User query to search for.
        page_size: Number of results to request (maps to ``pageSize``).
        max_chars: Budget for total snippet text kept across sources.
        timeout: Per-request timeout in seconds.
        retry_unquoted: When True (default) and the query contains quote
            characters, fall back to the unquoted query if the quoted search
            returns 0 results. Set False to disable the fallback.

    Returns:
        List of up to ``page_size`` GroundingSources, ordered by relevance.
    """
    invalid = _validate_args(config, query, page_size)
    if invalid:
        logger.warning("search_sources: %s", invalid)
        return []

    has_quotes = bool(_QUOTE_RE.search(query))
    if has_quotes:
        if not _has_searchable_content(query):
            logger.debug(
                "search_sources: query is only quote characters; nothing to search"
            )
            return []
        # Preserve exact-phrase semantics: search the quoted query first, and
        # only fall back to the unquoted form when it returns no results.
        first_query = _collapse_ws(query)
    else:
        first_query = _normalize_query(query)
    sources, zero_results = _search_sources_once(
        config, first_query, page_size, max_chars, timeout
    )
    if sources or not has_quotes or not retry_unquoted or not zero_results:
        return sources
    unquoted = _normalize_query(query)
    if unquoted == first_query:
        # Nothing to fall back to (e.g. a quotes-only query): the unquoted
        # form is identical, so a retry would be a duplicate search.
        return sources
    logger.debug("search_sources: quoted query returned 0 results; retrying unquoted")
    sources, _ = _search_sources_once(config, unquoted, page_size, max_chars, timeout)
    return sources


async def _asearch_sources_once(
    config: GreyCloudConfig,
    query: str,
    page_size: int,
    max_chars: int,
    timeout: float,
) -> Tuple[List[GroundingSource], bool]:
    """Async twin of :func:`_search_sources_once` (same contract)."""
    headers, auth_error = _build_headers(config)
    if auth_error:
        logger.warning("asearch_sources: %s", auth_error)
        return [], False

    datastore = getattr(config, "vertex_ai_search_datastore")
    # Read the flag once: the payload and the 400 log classification must
    # agree, and a divergent second read could misclassify the failure.
    spec = bool(getattr(config, "extractive_content_spec", False))
    last_exc: Optional[BaseException] = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            url = _search_url(datastore)
            payload = _search_payload(query, page_size, extractive_content_spec=spec)
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
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                _log_search_http_error(
                    "asearch_sources",
                    response,
                    extractive_requested=spec,
                )
                return [], False
            try:
                data = response.json()
                sources = _shape_results(data, max_chars)
            except ValueError as e:
                # Server-data problem (unparseable body, or a body whose shape
                # we can't interpret), not transient: do not retry.
                logger.warning(
                    "asearch_sources: invalid or malformed Discovery Engine "
                    "response (HTTP %s): %s",
                    response.status_code,
                    e,
                )
                return [], False
            return sources, not sources
        except Exception as e:  # noqa: BLE001 - must never raise to the caller
            last_exc = e
            logger.warning(
                "asearch_sources: attempt %d failed: %s; retrying",
                attempt + 1,
                e,
            )
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

    logger.warning(
        "asearch_sources: Discovery Engine search failed after %d attempts: %s",
        _MAX_ATTEMPTS,
        last_exc,
    )
    return [], False


async def asearch_sources(
    config: GreyCloudConfig,
    query: str,
    page_size: int = 5,
    max_chars: int = 8000,
    timeout: float = 30.0,
    retry_unquoted: bool = True,
) -> List[GroundingSource]:
    """Search the configured Discovery Engine datastore (async).

    Same contract as :func:`search_sources`: never raises, returns ``[]`` on
    any failure so the caller can generate without context. See
    :func:`search_sources` for the ``retry_unquoted`` behavior.
    """
    invalid = _validate_args(config, query, page_size)
    if invalid:
        logger.warning("asearch_sources: %s", invalid)
        return []

    has_quotes = bool(_QUOTE_RE.search(query))
    if has_quotes:
        if not _has_searchable_content(query):
            logger.debug(
                "asearch_sources: query is only quote characters; nothing to search"
            )
            return []
        first_query = _collapse_ws(query)
    else:
        first_query = _normalize_query(query)
    sources, zero_results = await _asearch_sources_once(
        config, first_query, page_size, max_chars, timeout
    )
    if sources or not has_quotes or not retry_unquoted or not zero_results:
        return sources
    unquoted = _normalize_query(query)
    if unquoted == first_query:
        return sources
    logger.debug("asearch_sources: quoted query returned 0 results; retrying unquoted")
    sources, _ = await _asearch_sources_once(
        config, unquoted, page_size, max_chars, timeout
    )
    return sources


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
        When you quote from the sources above, quote only from them, and cite
        each quoted passage with its [n] citation number in brackets.

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


def _invoke_grounding_callback(
    on_grounding: Callable[[List[GroundingSource]], None],
    sources: List[GroundingSource],
) -> None:
    """Invoke a sync ``on_grounding`` callback, never raising (proposal §5).

    A coroutine-function callback is driven to completion with ``asyncio.run``
    when no loop is running in this thread. When the sync client is used from
    inside a running loop, the coroutine is scheduled on that loop instead of
    being dropped, with failures logged from a done-callback — a scheduled
    callback exception must not surface as an unretrieved-task warning nor
    break generation.
    """
    try:
        result = on_grounding(sources)
        if result is not None and asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No loop in this thread: drive to completion inline.
                asyncio.run(result)
            else:

                def _log_task_failure(task: "asyncio.Task") -> None:
                    if task.cancelled() or task.exception() is None:
                        return
                    logger.warning("on_grounding callback failed: %s", task.exception())

                loop.create_task(result).add_done_callback(_log_task_failure)
    except Exception as e:  # noqa: BLE001 - callback must never break generation
        logger.warning("on_grounding callback failed: %s", e)


async def _ainvoke_grounding_callback(
    on_grounding: Callable[[List[GroundingSource]], None],
    sources: List[GroundingSource],
) -> None:
    """Async twin of :func:`_invoke_grounding_callback` (same contract).

    A coroutine-function callback is awaited; a plain function is called
    synchronously on the event loop. Never raises.
    """
    try:
        result = on_grounding(sources)
        if result is not None and asyncio.iscoroutine(result):
            await result
    except Exception as e:  # noqa: BLE001 - callback must never break generation
        logger.warning("on_grounding callback failed: %s", e)
