"""Unit tests for the Discovery Engine grounding module.

All tests mock the HTTP transport (``greycloud.grounding.requests.post`` and
``greycloud.grounding.httpx.AsyncClient``) — no live GCP calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from greycloud.config import GreyCloudConfig
from greycloud.grounding import (
    GroundingSource,
    asearch_sources,
    build_grounding_context,
    discovery_endpoint_for_datastore,
    search_sources,
    _clean_snippet,
    _MAX_ATTEMPTS,
    _normalize_query,
)

DATASTORE = (
    "projects/test-project/locations/us/collections/default_collection/"
    "dataStores/test-datastore"
)

OK_PAYLOAD = {
    "results": [
        {
            "document": {
                "derivedStructData": {
                    "title": "Guide_Is_This_Autism",
                    "link": "gs://bucket/Guide_Is_This_Autism.pdf",
                    "snippets": [
                        {"snippet": "<b>Autistic inertia</b> is difficult &nbsp; to manage."}
                    ],
                }
            }
        }
    ]
}


class FakeResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class BadJSONResponse(FakeResponse):
    """A 2xx response whose body is not valid JSON."""

    def json(self):
        raise ValueError("Expecting value: line 1 column 1")


@pytest.fixture
def grounding_config():
    """GreyCloudConfig pointing at a test datastore."""
    return GreyCloudConfig(
        project_id="test-project-id",
        location="us-east4",
        use_api_key=False,
        sa_email=None,
        vertex_ai_search_datastore=DATASTORE,
    )


def patch_async_client(result, side_effect=None):
    """Patch greycloud.grounding.httpx.AsyncClient with an AsyncMock-backed client."""
    client = MagicMock()
    if side_effect is not None:
        client.post = AsyncMock(side_effect=side_effect)
    else:
        client.post = AsyncMock(return_value=result)
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=client)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return patch("greycloud.grounding.httpx.AsyncClient", factory)


class TestDiscoveryEndpointForDatastore:
    def test_global(self):
        path = "projects/p/locations/global/collections/default_collection/dataStores/d"
        assert discovery_endpoint_for_datastore(path) == "https://discoveryengine.googleapis.com"

    def test_us(self):
        assert discovery_endpoint_for_datastore(DATASTORE) == "https://us-discoveryengine.googleapis.com"

    def test_eu(self):
        path = "projects/p/locations/eu/collections/default_collection/dataStores/d"
        assert discovery_endpoint_for_datastore(path) == "https://eu-discoveryengine.googleapis.com"

    def test_unknown_location_falls_back_to_global(self):
        path = "projects/p/locations/asia/collections/default_collection/dataStores/d"
        assert discovery_endpoint_for_datastore(path) == "https://discoveryengine.googleapis.com"

    def test_missing_location_falls_back_to_global(self):
        assert discovery_endpoint_for_datastore("projects/p/dataStores/d") == "https://discoveryengine.googleapis.com"


class TestNormalizeQuery:
    """_normalize_query is a pure helper: strip double quotes, collapse
    whitespace, keep apostrophes, never raise."""

    def test_quoted_title_quotes_removed_text_preserved(self):
        query = '"Is This Autism? A Guide for Clinicians and Everyone Else" autistic inertia'
        assert _normalize_query(query) == (
            "Is This Autism? A Guide for Clinicians and Everyone Else autistic inertia"
        )

    def test_section_15_evidence_regression(self):
        # Diagnosis 1.5: the quoted full title collapses the search to 0
        # results while the unquoted variant returns 5. Normalization must
        # turn the quoted form into exactly the unquoted form.
        quoted = '"Is This Autism? A Guide for Clinicians and Everyone Else" autistic inertia'
        unquoted = "Is This Autism? A Guide for Clinicians and Everyone Else autistic inertia"
        assert _normalize_query(quoted) == unquoted

    def test_internal_whitespace_collapsed_and_trimmed(self):
        assert _normalize_query("   Is   This   Autism   ") == "Is This Autism"
        assert _normalize_query("\t\n multi \n line \t") == "multi line"

    def test_no_quotes_is_no_op(self):
        query = "How should I explain autistic inertia to a parent of a child with autism?"
        assert _normalize_query(query) == query

    def test_apostrophes_and_single_quotes_preserved(self):
        assert _normalize_query("children's questions") == "children's questions"
        assert _normalize_query("'single quoted' term") == "'single quoted' term"

    def test_mixed_quotes_strip_only_double(self):
        assert _normalize_query('he said "it\'s fine"') == "he said it's fine"

    def test_only_quotes_returns_original(self):
        # Stripping all quotes leaves empty text; the conservative fallback
        # returns the original query unchanged (never an empty payload query).
        assert _normalize_query('"""') == '"""'

    def test_never_raises_on_anomaly(self):
        # A non-string input must be returned unchanged, not raise.
        assert _normalize_query(None) is None  # type: ignore[arg-type]
        assert _normalize_query(12345) == 12345  # type: ignore[arg-type]


class TestSearchSources:
    def test_request_shape_and_result_extraction(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({"Authorization": "Bearer tok"}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, OK_PAYLOAD)) as mock_post:
                sources = search_sources(grounding_config, "autistic inertia")

        assert len(sources) == 1
        source = sources[0]
        assert source.title == "Guide_Is_This_Autism"
        assert source.link == "gs://bucket/Guide_Is_This_Autism.pdf"
        assert source.snippet == "Autistic inertia is difficult to manage."
        assert source.index == 1

        url = mock_post.call_args[0][0]
        assert url == (
            "https://us-discoveryengine.googleapis.com/v1/"
            f"{DATASTORE}/servingConfigs/default_search:search"
        )
        body = mock_post.call_args[1]["json"]
        assert body["query"] == "autistic inertia"
        assert body["pageSize"] == 5
        assert body["contentSearchSpec"]["snippetSpec"]["returnSnippet"] is True
        assert mock_post.call_args[1]["headers"] == {"Authorization": "Bearer tok"}
        assert mock_post.call_args[1]["timeout"] == 30.0

    def test_quoted_query_stripped_in_payload(self, grounding_config):
        quoted = '"Is This Autism? A Guide for Clinicians and Everyone Else" autistic inertia'
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, {"results": []})) as mock_post:
                search_sources(grounding_config, quoted)
        body = mock_post.call_args[1]["json"]
        assert '"' not in body["query"]
        assert body["query"] == "Is This Autism? A Guide for Clinicians and Everyone Else autistic inertia"

    def test_unquoted_query_unchanged_in_payload(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, {"results": []})) as mock_post:
                search_sources(grounding_config, "autistic inertia")
        assert mock_post.call_args[1]["json"]["query"] == "autistic inertia"

    def test_debug_log_when_quotes_stripped(self, grounding_config, caplog):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, {"results": []})):
                with caplog.at_level("DEBUG", logger="greycloud.grounding"):
                    search_sources(grounding_config, '"quoted" term')
        assert any("stripped 2 quote character(s)" in r.message for r in caplog.records)

    def test_no_debug_log_when_no_quotes(self, grounding_config, caplog):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, {"results": []})):
                with caplog.at_level("DEBUG", logger="greycloud.grounding"):
                    search_sources(grounding_config, "autistic inertia")
        assert not any("quote character(s)" in r.message for r in caplog.records)

    def test_page_size_maps_to_pageSize(self, grounding_config):
        payload = {"results": []}
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, payload)) as mock_post:
                search_sources(grounding_config, "q", page_size=8)
        assert mock_post.call_args[1]["json"]["pageSize"] == 8

    def test_index_is_one_based(self, grounding_config):
        payload = {
            "results": [
                {"document": {"derivedStructData": {"title": "T1", "link": "L1", "snippets": [{"snippet": "one"}]}}},
                {"document": {"derivedStructData": {"title": "T2", "link": "L2", "snippets": [{"snippet": "two"}]}}},
            ]
        }
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, payload)):
                sources = search_sources(grounding_config, "q")
        assert [s.index for s in sources] == [1, 2]
        assert [s.title for s in sources] == ["T1", "T2"]

    def test_snippet_html_stripping_and_entity_decoding(self, grounding_config):
        payload = {
            "results": [
                {"document": {"derivedStructData": {
                    "title": "T", "link": "L",
                    "snippets": [{"snippet": "<b>Bold</b> &amp; <i>italic</i> &nbsp; text"}]}}}
            ]
        }
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, payload)):
                sources = search_sources(grounding_config, "q")
        assert sources[0].snippet == "Bold & italic text"

    def test_clean_snippet_direct(self):
        assert _clean_snippet("<b>Some</b> text &amp; more&nbsp;here") == "Some text & more here"
        assert _clean_snippet("&lt;b&gt;escaped&lt;/b&gt;") == "escaped"
        assert _clean_snippet(None) == ""

    def test_http_5xx_returns_empty(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(503, {}, "err")):
                with patch("greycloud.grounding.time.sleep"):
                    assert search_sources(grounding_config, "q") == []

    def test_non_2xx_returns_empty(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(403, {}, "forbidden")):
                assert search_sources(grounding_config, "q") == []

    def test_network_error_returns_empty(self, grounding_config):
        import requests

        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", side_effect=requests.ConnectionError("down")):
                with patch("greycloud.grounding.time.sleep"):
                    assert search_sources(grounding_config, "q") == []

    def test_malformed_json_returns_empty_without_retry(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=BadJSONResponse(200)) as mock_post:
                assert search_sources(grounding_config, "q") == []
                # Unparseable body is a server-data problem: single attempt, no retry.
                assert mock_post.call_count == 1

    def test_5xx_retries_then_empty(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(500, {}, "err")) as mock_post:
                with patch("greycloud.grounding.time.sleep"):
                    assert search_sources(grounding_config, "q") == []
                assert mock_post.call_count == _MAX_ATTEMPTS

    def test_no_backoff_sleep_on_final_attempt(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(500, {}, "err")):
                with patch("greycloud.grounding.time.sleep") as mock_sleep:
                    assert search_sources(grounding_config, "q") == []
                    assert mock_sleep.call_count == _MAX_ATTEMPTS - 1

    def test_retry_recovers_on_second_attempt(self, grounding_config):
        responses = [FakeResponse(503, {}, "err"), FakeResponse(200, OK_PAYLOAD)]
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", side_effect=responses) as mock_post:
                with patch("greycloud.grounding.time.sleep"):
                    sources = search_sources(grounding_config, "q")
        assert len(sources) == 1
        assert mock_post.call_count == 2

    def test_429_retries_then_empty(self, grounding_config):
        # 429 is the classic transient Discovery Engine throttle: retried like 5xx.
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(429, {}, "throttled")) as mock_post:
                with patch("greycloud.grounding.time.sleep"):
                    assert search_sources(grounding_config, "q") == []
                assert mock_post.call_count == _MAX_ATTEMPTS

    def test_429_recovers_on_second_attempt(self, grounding_config):
        responses = [FakeResponse(429, {}, "throttled"), FakeResponse(200, OK_PAYLOAD)]
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", side_effect=responses) as mock_post:
                with patch("greycloud.grounding.time.sleep"):
                    sources = search_sources(grounding_config, "q")
        assert len(sources) == 1
        assert mock_post.call_count == 2

    def test_snippets_as_dict_treated_as_no_snippet(self, grounding_config):
        # Malformed-shape response: ``snippets`` is a dict, not a list. Should
        # yield an empty snippet (not raise, not burn a retry attempt).
        payload = {
            "results": [
                {"document": {"derivedStructData": {
                    "title": "T", "link": "L", "snippets": {"snippet": "not a list"}}}}
            ]
        }
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, payload)) as mock_post:
                sources = search_sources(grounding_config, "q")
        assert len(sources) == 1
        assert sources[0].title == "T"
        assert sources[0].link == "L"
        assert sources[0].snippet == ""
        assert mock_post.call_count == 1  # no retries on malformed shape

    def test_invalid_query_returns_empty_without_request(self, grounding_config):
        with patch("greycloud.grounding.requests.post") as mock_post:
            assert search_sources(grounding_config, "") == []
            assert search_sources(grounding_config, "   ") == []
            assert search_sources(grounding_config, None) == []
            assert search_sources(grounding_config, 12345) == []
            assert mock_post.call_count == 0

    def test_missing_datastore_returns_empty_without_request(self):
        config = GreyCloudConfig(project_id="p", location="us-east4", vertex_ai_search_datastore=None)
        with patch("greycloud.grounding.requests.post") as mock_post:
            assert search_sources(config, "q") == []
            assert mock_post.call_count == 0

    def test_wrong_typed_datastore_returns_empty_without_request(self):
        config = GreyCloudConfig(project_id="p", location="us-east4", vertex_ai_search_datastore=12345)
        with patch("greycloud.grounding.requests.post") as mock_post:
            assert search_sources(config, "q") == []
            assert mock_post.call_count == 0

    def test_wrong_typed_page_size_returns_empty_without_request(self, grounding_config):
        with patch("greycloud.grounding.requests.post") as mock_post:
            assert search_sources(grounding_config, "q", page_size="five") == []
            assert mock_post.call_count == 0

    def test_auth_failure_returns_empty_without_request(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=(None, "no creds")):
            with patch("greycloud.grounding.requests.post") as mock_post:
                assert search_sources(grounding_config, "q") == []
                assert mock_post.call_count == 0

    def test_api_key_header_path(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({"x-goog-api-key": "key-abc"}, None)):
            with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, OK_PAYLOAD)) as mock_post:
                search_sources(grounding_config, "q")
        assert mock_post.call_args[1]["headers"] == {"x-goog-api-key": "key-abc"}


class TestCredentialRefresh:
    """_build_headers must refresh only when the token is missing/expired.

    google.auth.default() returns a module-cached credentials object, and each
    impersonated-credential refresh costs one IAM call; refreshing a still-valid
    token on every search would burn a metadata-server hit / refresh-token grant
    per turn.
    """

    class FakeCreds:
        def __init__(self, token="valid", expired=False):
            self.token = token
            self.expired = expired
            self.refresh_calls = []

        def refresh(self, request):
            self.refresh_calls.append(request)

    def test_no_refresh_when_token_valid(self, grounding_config):
        creds = self.FakeCreds(token="valid", expired=False)
        with patch("greycloud.grounding.get_credentials", return_value=creds):
            with patch("google.auth.transport.requests.Request") as mock_request:
                with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, OK_PAYLOAD)):
                    sources = search_sources(grounding_config, "q")

        assert len(sources) == 1
        assert creds.refresh_calls == []
        mock_request.assert_not_called()

    def test_refresh_when_expired(self, grounding_config):
        creds = self.FakeCreds(token="stale", expired=True)
        with patch("greycloud.grounding.get_credentials", return_value=creds):
            with patch("google.auth.transport.requests.Request") as mock_request:
                with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, OK_PAYLOAD)):
                    sources = search_sources(grounding_config, "q")

        assert len(sources) == 1
        assert len(creds.refresh_calls) == 1
        assert creds.refresh_calls[0] is mock_request.return_value

    def test_refresh_when_token_missing(self, grounding_config):
        creds = self.FakeCreds(token=None, expired=True)
        with patch("greycloud.grounding.get_credentials", return_value=creds):
            with patch("google.auth.transport.requests.Request"):
                with patch("greycloud.grounding.requests.post", return_value=FakeResponse(200, OK_PAYLOAD)):
                    sources = search_sources(grounding_config, "q")

        assert len(sources) == 1
        assert len(creds.refresh_calls) == 1


class TestBuildGroundingContext:
    def test_renders_numbered_sources(self):
        sources = [
            GroundingSource(title="Guide_Is_This_Autism", link="gs://bucket/Guide.pdf", snippet="Passage one.", index=1),
            GroundingSource(title="Doc2", link="gs://bucket/doc2.pdf", snippet="Passage two.", index=2),
        ]
        ctx = build_grounding_context(sources)
        assert ctx.startswith("<grounding_sources>")
        assert "[1] (Guide_Is_This_Autism — gs://bucket/Guide.pdf)" in ctx
        assert '"Passage one."' in ctx
        assert "[2] (Doc2 — gs://bucket/doc2.pdf)" in ctx
        assert '"Passage two."' in ctx
        assert "</grounding_sources>" in ctx
        assert "[n]" in ctx  # instruction tells the model to cite [n]

    def test_empty_sources_returns_empty_string(self):
        assert build_grounding_context([]) == ""

    def test_max_chars_cap_never_exceeded(self):
        sources = [
            GroundingSource(title="T", link="L", snippet="y" * 100, index=i) for i in range(1, 6)
        ]
        for cap in (0, 10, 100, 1000, 8000):
            ctx = build_grounding_context(sources, max_chars=cap)
            assert len(ctx) <= cap, f"cap={cap} exceeded"

    def test_later_sources_truncated_first(self):
        sources = [
            GroundingSource(title="T1", link="L1", snippet="a" * 100, index=1),
            GroundingSource(title="T2", link="L2", snippet="b" * 100, index=2),
        ]
        ctx = build_grounding_context(sources, max_chars=250)
        assert len(ctx) <= 250
        # The cap is tight enough that even the first snippet is truncated,
        # but truncation always happens from the end: later sources drop first.
        assert "a" * 10 in ctx  # first source's snippet retained (as a prefix)
        assert "b" * 10 not in ctx  # second source's snippet fully dropped


class TestASearchSources:
    @pytest.mark.asyncio
    async def test_success_and_request_shape(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({"Authorization": "Bearer tok"}, None)):
            with patch_async_client(FakeResponse(200, OK_PAYLOAD)) as factory:
                sources = await asearch_sources(grounding_config, "autistic inertia")

        assert len(sources) == 1
        assert sources[0].title == "Guide_Is_This_Autism"
        assert sources[0].snippet == "Autistic inertia is difficult to manage."
        assert sources[0].index == 1

        post_mock = factory.return_value.__aenter__.return_value.post
        url = post_mock.call_args[0][0]
        assert url == (
            "https://us-discoveryengine.googleapis.com/v1/"
            f"{DATASTORE}/servingConfigs/default_search:search"
        )
        body = post_mock.call_args[1]["json"]
        assert body["query"] == "autistic inertia"
        assert body["pageSize"] == 5
        assert body["contentSearchSpec"]["snippetSpec"]["returnSnippet"] is True
        assert post_mock.call_args[1]["headers"] == {"Authorization": "Bearer tok"}

    @pytest.mark.asyncio
    async def test_quoted_query_stripped_in_payload(self, grounding_config):
        quoted = '"Is This Autism? A Guide for Clinicians and Everyone Else" autistic inertia'
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch_async_client(FakeResponse(200, {"results": []})) as factory:
                await asearch_sources(grounding_config, quoted)
        post_mock = factory.return_value.__aenter__.return_value.post
        body = post_mock.call_args[1]["json"]
        assert '"' not in body["query"]
        assert body["query"] == "Is This Autism? A Guide for Clinicians and Everyone Else autistic inertia"

    @pytest.mark.asyncio
    async def test_unquoted_query_unchanged_in_payload(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch_async_client(FakeResponse(200, {"results": []})) as factory:
                await asearch_sources(grounding_config, "autistic inertia")
        post_mock = factory.return_value.__aenter__.return_value.post
        assert post_mock.call_args[1]["json"]["query"] == "autistic inertia"

    @pytest.mark.asyncio
    async def test_5xx_returns_empty(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch_async_client(FakeResponse(503, {}, "err")) as factory:
                with patch("greycloud.grounding.asyncio.sleep"):
                    assert await asearch_sources(grounding_config, "q") == []
        assert factory.return_value.__aenter__.return_value.post.call_count == _MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_429_returns_empty(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch_async_client(FakeResponse(429, {}, "throttled")) as factory:
                with patch("greycloud.grounding.asyncio.sleep"):
                    assert await asearch_sources(grounding_config, "q") == []
        assert factory.return_value.__aenter__.return_value.post.call_count == _MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_non_2xx_returns_empty(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch_async_client(FakeResponse(404, {}, "nf")):
                assert await asearch_sources(grounding_config, "q") == []

    @pytest.mark.asyncio
    async def test_network_error_returns_empty(self, grounding_config):
        import httpx

        async def boom(*args, **kwargs):
            raise httpx.ConnectError("down")

        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch_async_client(None, side_effect=boom):
                assert await asearch_sources(grounding_config, "q") == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty_without_retry(self, grounding_config):
        with patch("greycloud.grounding._build_headers", return_value=({}, None)):
            with patch_async_client(BadJSONResponse(200)) as factory:
                assert await asearch_sources(grounding_config, "q") == []
            assert factory.return_value.__aenter__.return_value.post.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_args_returns_empty_without_request(self, grounding_config):
        with patch("greycloud.grounding.httpx.AsyncClient") as mock_client:
            assert await asearch_sources(grounding_config, "") == []
            assert await asearch_sources(grounding_config, None) == []
            assert await asearch_sources(grounding_config, 42) == []
            assert mock_client.call_count == 0

    @pytest.mark.asyncio
    async def test_wrong_typed_datastore_returns_empty_without_request(self):
        config = GreyCloudConfig(project_id="p", location="us-east4", vertex_ai_search_datastore=12345)
        with patch("greycloud.grounding.httpx.AsyncClient") as mock_client:
            assert await asearch_sources(config, "q") == []
            assert mock_client.call_count == 0
