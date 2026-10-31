"""
Gemini 4 Argon Contract and End-to-End Verification Tests.

Comprehensive contract tests simulating Gemini 4 Argon workflows:
1. Argon config building: verifies unsupported penalties (frequency_penalty,
   presence_penalty) and sampling parameters (temperature, top_p, top_k)
   are stripped/omitted, thinking_config is built with MEDIUM and
   include_thoughts=True, custom levels (LOW, HIGH) are supported, and
   MINIMAL or thinking_budget are rejected with actionable errors.
2. Dual-mode streaming:
   - Clean text default: parts with thought=True are filtered out and only
     candidate text strings are yielded without premature stream breaks.
   - return_events=True: {"thought": ...} and {"text": ...} events are yielded
     matching SSE event schema.
   - on_thought callback: thought chunks fire sync/async callbacks in real time,
     and callback errors never disrupt streaming.
3. Timeout pass-through: per-call request_timeout overrides config defaults
   and passes timeout in milliseconds to HttpOptions across unary and stream.
4. Implicit caching enforcement: explicit cached_content and cache creation APIs
   raise ValueError with guidance on automatic implicit durable caching in Vertex AI.
5. Rate limiter thoughts reconciliation: surplus reasoning tokens from
   usage_metadata are debited from the rate limiter upon completion.
6. Helper _extract_total_tokens: verifies dictionary and object parsing with
   camelCase, snake_case, and component-sum fallback coverage.

All tests adhere to domain naming conventions (distributed systems, consensus
protocols, replicated state machines) without clinical or medical terms.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from google.genai import types

from greycloud import (
    GreyCloudConfig,
    GreyCloudClient,
    GreyCloudAsyncClient,
    GreyCloudCache,
    GreyCloudBatch,
    VertexRateLimiter,
)
from greycloud.cache import MIN_TOKENS_BY_MODEL
from greycloud.config import is_gemini_4
from greycloud.auth import create_client

# =====================================================================
# Helpers and Fixtures
# =====================================================================


def _make_chunk(text: str, thought: bool = False, total_tokens: int | None = None):
    """Build a GenerateContentResponse chunk with candidate parts."""
    part = types.Part(text=text, thought=thought)
    content = types.Content(parts=[part], role="model")
    candidate = types.Candidate(content=content)
    meta = None
    if total_tokens is not None:
        meta = types.GenerateContentResponseUsageMetadata(
            total_token_count=total_tokens,
            prompt_token_count=30,
            candidates_token_count=70,
            thoughts_token_count=max(0, total_tokens - 100),
        )
    return types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=meta,
    )


async def _mock_stream_awaitable(*chunks):
    """Coroutine that yields an async generator of chunks for async client."""

    async def _gen():
        for c in chunks:
            yield c

    return _gen()


@pytest.fixture
def consensus_contents():
    """Generic distributed consensus request contents."""
    return [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "Explain the log compaction and snapshot protocol "
                        "in Raft consensus."
                    )
                )
            ],
        )
    ]


@pytest.fixture
def argon_config():
    """Config configured for Gemini 4 Argon with thinking parameters."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test-project-id"
        return GreyCloudConfig(
            project_id="test-consensus-cluster",
            location="us-east4",
            model="gemini-4-argon",
            temperature=0.7,
            top_p=0.9,
            thinking_level="MEDIUM",
            include_thoughts=True,
            request_timeout=1200,
            use_api_key=False,
            sa_email=None,
        )


# =====================================================================
# 1. Argon Configuration Building Contract Tests
# =====================================================================


class TestGemini4ConfigContract:
    """Verifies request configuration sanitization for Gemini 4 Argon models."""

    def test_is_gemini_4_classification(self):
        """Verifies is_gemini_4 pattern matching across models and families."""
        assert is_gemini_4("gemini-4-argon") is True
        assert is_gemini_4("gemini-4-carbon") is True
        assert is_gemini_4("projects/123/models/gemini-4-argon") is True
        assert is_gemini_4("gemini-3.8-flash") is False
        assert is_gemini_4("gemini-2.5-pro") is False
        assert is_gemini_4("") is False
        assert is_gemini_4(None) is False  # type: ignore[arg-type]

    def test_argon_sync_config_building_strips_penalties_and_sampling(
        self, argon_config, consensus_contents
    ):
        """Sync client strips sampling and penalties, setting MEDIUM thinking config."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            client = GreyCloudClient(argon_config)

            cfg = client._build_generate_config(
                model="gemini-4-argon",
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                frequency_penalty=0.4,
                presence_penalty=0.6,
            )

            # Sampling and penalty parameters must be omitted/None
            assert cfg.temperature is None
            assert cfg.top_p is None
            assert getattr(cfg, "top_k", None) is None
            assert getattr(cfg, "frequency_penalty", None) is None
            assert getattr(cfg, "presence_penalty", None) is None

            # Thinking config must default to MEDIUM with include_thoughts=True
            assert cfg.thinking_config is not None
            assert (
                cfg.thinking_config.thinking_level == "MEDIUM"
                or cfg.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
            )
            assert cfg.thinking_config.include_thoughts is True

    def test_argon_async_config_building_strips_penalties_and_sampling(
        self, argon_config, consensus_contents
    ):
        """Async client strips sampling and penalties identically for Gemini 4."""
        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            async_client = GreyCloudAsyncClient(argon_config)

            cfg = async_client._build_generate_config(
                model="gemini-4-argon",
                temperature=0.2,
                top_p=0.8,
                top_k=20,
                frequency_penalty=0.1,
                presence_penalty=0.2,
            )

            assert cfg.temperature is None
            assert cfg.top_p is None
            assert getattr(cfg, "top_k", None) is None
            assert getattr(cfg, "frequency_penalty", None) is None
            assert getattr(cfg, "presence_penalty", None) is None
            assert cfg.thinking_config is not None
            assert (
                cfg.thinking_config.thinking_level == "MEDIUM"
                or cfg.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
            )
            assert cfg.thinking_config.include_thoughts is True

    @pytest.mark.parametrize("level", ["LOW", "HIGH"])
    def test_argon_config_custom_thinking_levels(self, argon_config, level):
        """Custom valid thinking levels (LOW, HIGH) are preserved."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)
            cfg = client._build_generate_config(
                model="gemini-4-argon", thinking_level=level
            )
            expected_enum = getattr(types.ThinkingLevel, level, level)
            assert (
                cfg.thinking_config.thinking_level == level
                or cfg.thinking_config.thinking_level == expected_enum
            )

    def test_argon_config_rejects_minimal_thinking_level(self, argon_config):
        """Gemini 4 rejects thinking_level='MINIMAL'."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)
            with pytest.raises(
                ValueError, match="does not support thinking_level='MINIMAL'"
            ):
                client._build_generate_config(
                    model="gemini-4-argon", thinking_level="MINIMAL"
                )

    def test_argon_config_rejects_invalid_thinking_level(self, argon_config):
        """Gemini 4 rejects unrecognized thinking levels."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)
            with pytest.raises(ValueError, match="Invalid thinking_level 'EXTREME'"):
                client._build_generate_config(
                    model="gemini-4-argon", thinking_level="EXTREME"
                )

    def test_argon_config_rejects_thinking_budget(self, argon_config):
        """Gemini 4 rejects thinking_budget in kwargs or thinking_config."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)

            with pytest.raises(
                ValueError, match="does not support thinking_budget; use thinking_level"
            ):
                client._build_generate_config(
                    model="gemini-4-argon", thinking_budget=2048
                )

            with pytest.raises(
                ValueError, match="does not support thinking_budget; use thinking_level"
            ):
                client._build_generate_config(
                    model="gemini-4-argon", thinkingBudget=2048
                )

            with pytest.raises(
                ValueError, match="does not support thinking_budget; use thinking_level"
            ):
                client._build_generate_config(
                    model="gemini-4-argon",
                    thinking_config={"thinking_budget": 1024},
                )

    def test_argon_config_populates_safety_system_and_seed(self, argon_config):
        """Gemini 4 preserves safety settings, system instructions, and seed."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)
            client.config.seed = 42
            cfg = client._build_generate_config(
                model="gemini-4-argon",
                system_instruction="Coordinate Paxos state machine replication.",
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_LOW_AND_ABOVE",
                    }
                ],
            )
            assert cfg.seed == 42
            assert len(cfg.system_instruction) == 1
            assert (
                cfg.system_instruction[0].text
                == "Coordinate Paxos state machine replication."
            )
            assert len(cfg.safety_settings) == 1
            assert (
                cfg.safety_settings[0].category == "HARM_CATEGORY_DANGEROUS_CONTENT"
                or cfg.safety_settings[0].category.name
                == "HARM_CATEGORY_DANGEROUS_CONTENT"
            )

    @patch("greycloud.batch.GCS_AVAILABLE", True)
    def test_argon_batch_serialization_omits_sampling_and_penalties(self, argon_config):
        """Batch JSONL generator omits sampling and penalty fields for Gemini 4."""
        requests = [
            types.InlinedRequest(
                model="gemini-4-argon",
                contents=[
                    {"role": "user", "parts": [{"text": "Explain vector clocks"}]}
                ],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=1000,
                ),
            ),
            types.InlinedRequest(
                model="gemini-4-argon",
                contents=[
                    {"role": "user", "parts": [{"text": "Explain two-phase commit"}]}
                ],
                config={
                    "temperature": 0.5,
                    "topP": 0.8,
                    "topK": 25,
                    "frequencyPenalty": 0.3,
                    "presencePenalty": 0.3,
                    "maxOutputTokens": 500,
                    "thinkingConfig": {"thinkingLevel": "HIGH", "thinkingBudget": 500},
                },
            ),
            types.InlinedRequest(
                model="gemini-3.8-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": "Explain gossip protocols"}],
                    }
                ],
                config={"temperature": 0.4, "topP": 0.85},
            ),
        ]

        with patch("greycloud.batch.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("greycloud.batch.storage") as mock_storage:
                mock_storage_client = MagicMock()
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_storage.Client.return_value = mock_storage_client
                mock_storage_client.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob

                batch = GreyCloudBatch(argon_config)
                batch.upload_batch_requests_to_gcs(
                    batch_requests=requests, bucket_name="distributed-consensus-bucket"
                )

                uploaded = mock_blob.upload_from_string.call_args[0][0]
                lines = uploaded.strip().split("\n")
                assert len(lines) == 3

                # Line 0: Gemini 4 InlinedRequest with GenerateContentConfig
                req0 = json.loads(lines[0])["request"]
                cfg0 = req0["generationConfig"]
                assert "temperature" not in cfg0
                assert "topP" not in cfg0
                assert "topK" not in cfg0
                assert cfg0["maxOutputTokens"] == 1000

                # Line 1: Gemini 4 InlinedRequest with dict config
                req1 = json.loads(lines[1])["request"]
                cfg1 = req1["generationConfig"]
                assert "temperature" not in cfg1
                assert "topP" not in cfg1
                assert "topK" not in cfg1
                assert "frequencyPenalty" not in cfg1
                assert "presencePenalty" not in cfg1
                assert cfg1["maxOutputTokens"] == 500
                assert "thinkingConfig" in cfg1
                assert "thinkingBudget" not in cfg1["thinkingConfig"]

                # Line 2: Gemini 3.x preserves sampling parameters
                req2 = json.loads(lines[2])["request"]
                cfg2 = req2["generationConfig"]
                assert cfg2["temperature"] == 0.4
                assert cfg2["topP"] == 0.85


# =====================================================================
# 2. Dual-Mode Streaming Contract Tests
# =====================================================================


class TestGemini4StreamingContract:
    """Verifies clean text streaming, event dict streaming, and on_thought callbacks."""

    def test_dual_mode_streaming_clean_text_default_sync(
        self, argon_config, consensus_contents
    ):
        """Default streaming skips thought=True parts, yielding only final text."""
        chunk1 = _make_chunk(
            "Analyzing distributed consensus quorum invariants...", thought=True
        )
        chunk2 = _make_chunk(
            "Raft cluster leader handles log replication.", thought=False
        )
        chunk3 = _make_chunk(" Follower nodes append entries.", thought=False)

        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.models.generate_content_stream.return_value = [
                chunk1,
                chunk2,
                chunk3,
            ]
            mock_create.return_value = mock_client

            client = GreyCloudClient(argon_config)
            yielded = list(client.generate_content_stream(consensus_contents))

            # Thinking chunks filtered out; candidate texts intact
            assert yielded == [
                "Raft cluster leader handles log replication.",
                " Follower nodes append entries.",
            ]

    @pytest.mark.asyncio
    async def test_dual_mode_streaming_clean_text_default_async(
        self, argon_config, consensus_contents
    ):
        """Async client default streaming yields only final text strings."""
        chunk1 = _make_chunk("Validating Paxos ballot numbers...", thought=True)
        chunk2 = _make_chunk(
            "Phase 1a Prepare messages broadcast to acceptors.", thought=False
        )

        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = MagicMock(
                side_effect=lambda **kw: _mock_stream_awaitable(chunk1, chunk2)
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            yielded = [
                item
                async for item in async_client.generate_content_stream(
                    consensus_contents
                )
            ]

            assert yielded == ["Phase 1a Prepare messages broadcast to acceptors."]

    def test_dual_mode_streaming_return_events_sync(
        self, argon_config, consensus_contents
    ):
        """return_events=True yields structured SSE event dictionaries."""
        chunk1 = _make_chunk("Formulating Byzantine fault proof...", thought=True)
        chunk2 = _make_chunk(
            "System tolerates f faulty nodes given 3f+1 replicas.", thought=False
        )

        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]
            mock_create.return_value = mock_client

            client = GreyCloudClient(argon_config)
            events = list(
                client.generate_content_stream(consensus_contents, return_events=True)
            )

            assert events == [
                {"thought": "Formulating Byzantine fault proof..."},
                {"text": "System tolerates f faulty nodes given 3f+1 replicas."},
            ]

    @pytest.mark.asyncio
    async def test_dual_mode_streaming_return_events_async(
        self, argon_config, consensus_contents
    ):
        """Async return_events=True yields structured SSE event dictionaries."""
        chunk1 = _make_chunk("Evaluating gossip dissemination rate...", thought=True)
        chunk2 = _make_chunk(
            "Anti-entropy epidemic protocol converges in O(log N).", thought=False
        )

        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = MagicMock(
                side_effect=lambda **kw: _mock_stream_awaitable(chunk1, chunk2)
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            events = [
                e
                async for e in async_client.generate_content_stream(
                    consensus_contents, return_events=True
                )
            ]

            assert events == [
                {"thought": "Evaluating gossip dissemination rate..."},
                {"text": "Anti-entropy epidemic protocol converges in O(log N)."},
            ]

    def test_streaming_on_thought_callback_sync(self, argon_config, consensus_contents):
        """Sync on_thought callback receives thought chunks in real time."""
        thought_events = []

        def handle_thought(t: str):
            thought_events.append(t)

        chunk1 = _make_chunk("Checking vector clock partial ordering...", thought=True)
        chunk2 = _make_chunk("Causal delivery satisfied.", thought=False)

        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]
            mock_create.return_value = mock_client

            client = GreyCloudClient(argon_config)
            yielded = list(
                client.generate_content_stream(
                    consensus_contents, on_thought=handle_thought
                )
            )

            assert thought_events == ["Checking vector clock partial ordering..."]
            assert yielded == ["Causal delivery satisfied."]

    @pytest.mark.asyncio
    async def test_streaming_on_thought_callback_async_coroutine(
        self, argon_config, consensus_contents
    ):
        """Async on_thought supports both coroutine and regular function callbacks."""
        coro_thoughts = []

        async def async_on_thought(t: str):
            coro_thoughts.append(t)

        chunk1 = _make_chunk(
            "Verifying state machine replication order...", thought=True
        )
        chunk2 = _make_chunk("Log entry committed at index 42.", thought=False)

        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = MagicMock(
                side_effect=lambda **kw: _mock_stream_awaitable(chunk1, chunk2)
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            yielded = [
                item
                async for item in async_client.generate_content_stream(
                    consensus_contents, on_thought=async_on_thought
                )
            ]

            assert coro_thoughts == ["Verifying state machine replication order..."]
            assert yielded == ["Log entry committed at index 42."]

    def test_streaming_on_thought_callback_error_resilience(
        self, argon_config, consensus_contents
    ):
        """Callback exceptions are logged at WARNING and streaming continues."""

        def exploding_callback(t: str):
            raise RuntimeError("Diagnostic connection interrupted")

        chunk1 = _make_chunk("Internal thought stream", thought=True)
        chunk2 = _make_chunk("Final resilient response", thought=False)

        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]
            mock_create.return_value = mock_client

            client = GreyCloudClient(argon_config)
            yielded = list(
                client.generate_content_stream(
                    consensus_contents, on_thought=exploding_callback
                )
            )
            assert yielded == ["Final resilient response"]

    def test_streaming_handles_chunk_level_text_fallback_sync(
        self, argon_config, consensus_contents
    ):
        """Fallback to chunk.text when part.text is not a string."""
        part = MagicMock()
        part.thought = False
        part.text = MagicMock()  # Non-string mock object triggers fallback

        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]
        chunk.text = "Fallback text from chunk root"

        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.models.generate_content_stream.return_value = [chunk]
            mock_create.return_value = mock_client

            client = GreyCloudClient(argon_config)
            yielded = list(client.generate_content_stream(consensus_contents))
            assert yielded == ["Fallback text from chunk root"]

    @pytest.mark.asyncio
    async def test_streaming_handles_chunk_level_text_fallback_async(
        self, argon_config, consensus_contents
    ):
        """Fallback to chunk.text in async streaming when part.text is not a string."""
        part = MagicMock()
        part.thought = False
        part.text = MagicMock()  # Non-string triggers fallback

        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]
        chunk.text = "Async fallback text from chunk root"
        chunk.usage_metadata = None

        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = MagicMock(
                side_effect=lambda **kw: _mock_stream_awaitable(chunk)
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            yielded = [
                item
                async for item in async_client.generate_content_stream(
                    consensus_contents
                )
            ]
            assert yielded == ["Async fallback text from chunk root"]


# =====================================================================
# 3. Timeout Pass-Through Contract Tests
# =====================================================================


class TestGemini4TimeoutContract:
    """Verifies per-call request_timeout overrides propagating to HttpOptions in ms."""

    def test_create_client_http_options_timeout_configuration(self):
        """create_client configures types.HttpOptions with milliseconds timeout."""
        with patch("greycloud.auth.get_credentials"):
            with patch("greycloud.auth.genai.Client") as mock_genai_cls:
                # Default request_timeout=1200 -> 1200000 ms
                create_client(
                    project_id="test-cluster",
                    location="us-east4",
                )
                http_opts = mock_genai_cls.call_args[1]["http_options"]
                assert http_opts.timeout == 1200000
                assert http_opts.api_version == "v1beta1"

        with patch("greycloud.auth.get_credentials"):
            with patch("greycloud.auth.genai.Client") as mock_genai_cls:
                # Custom request_timeout=45 -> 45000 ms
                create_client(
                    project_id="test-cluster",
                    location="us-east4",
                    request_timeout=45,
                )
                http_opts = mock_genai_cls.call_args[1]["http_options"]
                assert http_opts.timeout == 45000

        with patch("greycloud.auth.get_credentials"):
            with patch("greycloud.auth.genai.Client") as mock_genai_cls:
                # request_timeout=None -> None
                create_client(
                    project_id="test-cluster",
                    location="us-east4",
                    request_timeout=None,
                )
                http_opts = mock_genai_cls.call_args[1]["http_options"]
                assert http_opts.timeout is None

    def test_sync_generate_content_per_call_timeout(
        self, argon_config, consensus_contents
    ):
        """Per-call request_timeout overrides config in unary generate_content."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = MagicMock(
                text="Consensus reached"
            )
            mock_create.return_value = mock_client

            client = GreyCloudClient(argon_config)
            client.generate_content(consensus_contents, request_timeout=35)

            call_cfg = mock_client.models.generate_content.call_args[1]["config"]
            assert call_cfg.http_options is not None
            assert call_cfg.http_options.timeout == 35000

    def test_sync_generate_content_stream_per_call_timeout(
        self, argon_config, consensus_contents
    ):
        """Per-call request_timeout overrides config in sync stream."""
        with patch("greycloud.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.models.generate_content_stream.return_value = []
            mock_create.return_value = mock_client

            client = GreyCloudClient(argon_config)
            list(client.generate_content_stream(consensus_contents, request_timeout=25))

            call_cfg = mock_client.models.generate_content_stream.call_args[1]["config"]
            assert call_cfg.http_options is not None
            assert call_cfg.http_options.timeout == 25000

    @pytest.mark.asyncio
    async def test_async_generate_content_per_call_timeout(
        self, argon_config, consensus_contents
    ):
        """Per-call request_timeout overrides config in async generate_content."""
        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=MagicMock(text="Log replicated")
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            await async_client.generate_content(consensus_contents, request_timeout=50)

            call_cfg = mock_client.aio.models.generate_content.call_args[1]["config"]
            assert call_cfg.http_options is not None
            assert call_cfg.http_options.timeout == 50000

    @pytest.mark.asyncio
    async def test_async_generate_content_stream_per_call_timeout(
        self, argon_config, consensus_contents
    ):
        """Per-call request_timeout overrides config in async stream."""
        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = MagicMock(
                side_effect=lambda **kw: _mock_stream_awaitable()
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            async for _ in async_client.generate_content_stream(
                consensus_contents, request_timeout=65
            ):
                pass

            call_cfg = mock_client.aio.models.generate_content_stream.call_args[1][
                "config"
            ]
            assert call_cfg.http_options is not None
            assert call_cfg.http_options.timeout == 65000

    def test_build_generate_config_kwargs_timeout_and_http_options(self, argon_config):
        """_build_generate_config handles kwargs request_timeout and http_options."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)

            cfg1 = client._build_generate_config(
                model="gemini-4-argon", request_timeout=40
            )
            assert cfg1.http_options.timeout == 40000

            custom_opts = types.HttpOptions(timeout=80000)
            cfg2 = client._build_generate_config(
                model="gemini-4-argon", http_options=custom_opts
            )
            assert cfg2.http_options.timeout == 80000


# =====================================================================
# 4. Implicit Caching Enforcement Contract Tests
# =====================================================================


class TestGemini4ImplicitCachingContract:
    """Verifies rejection of explicit cache creation and cached_content on Gemini 4."""

    def test_sync_generate_content_rejects_cached_content(
        self, argon_config, consensus_contents
    ):
        """Sync generate_content raises ValueError when cached_content is provided."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)
            with pytest.raises(
                ValueError, match="automatic implicit durable caching in Vertex AI"
            ):
                client.generate_content(
                    consensus_contents,
                    model="gemini-4-argon",
                    cached_content="cachedContents/paxos-spec-cache-001",
                )

    def test_sync_generate_content_stream_rejects_cached_content(
        self, argon_config, consensus_contents
    ):
        """Sync stream raises ValueError when cached_content is provided."""
        with patch("greycloud.client.create_client"):
            client = GreyCloudClient(argon_config)
            with pytest.raises(
                ValueError, match="automatic implicit durable caching in Vertex AI"
            ):
                list(
                    client.generate_content_stream(
                        consensus_contents,
                        model="gemini-4-argon",
                        cached_content="cachedContents/paxos-spec-cache-001",
                    )
                )

    @pytest.mark.asyncio
    async def test_async_generate_content_rejects_cached_content(
        self, argon_config, consensus_contents
    ):
        """Async generate_content raises ValueError when cached_content is provided."""
        with patch("greycloud.async_client.create_client"):
            async_client = GreyCloudAsyncClient(argon_config)
            with pytest.raises(
                ValueError, match="automatic implicit durable caching in Vertex AI"
            ):
                await async_client.generate_content(
                    consensus_contents,
                    model="gemini-4-argon",
                    cached_content="cachedContents/raft-spec-cache-002",
                )

    @pytest.mark.asyncio
    async def test_async_generate_content_stream_rejects_cached_content(
        self, argon_config, consensus_contents
    ):
        """Async stream raises ValueError when cached_content is provided."""
        with patch("greycloud.async_client.create_client"):
            async_client = GreyCloudAsyncClient(argon_config)
            with pytest.raises(
                ValueError, match="automatic implicit durable caching in Vertex AI"
            ):
                async for _ in async_client.generate_content_stream(
                    consensus_contents,
                    model="gemini-4-argon",
                    cached_content="cachedContents/raft-spec-cache-002",
                ):
                    pass

    def test_cache_client_create_cache_rejects_gemini_4(self, argon_config):
        """create_cache rejects Gemini 4 models with implicit caching error."""
        with patch("greycloud.cache.create_client"):
            cache = GreyCloudCache(argon_config)
            with pytest.raises(
                ValueError, match="automatic implicit durable caching in Vertex AI"
            ):
                cache.create_cache(
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text="Distributed storage spec")
                            ],
                        )
                    ],
                    model="gemini-4-argon",
                )

    def test_cache_client_create_from_text_rejects_gemini_4(self, argon_config):
        """create_cache_from_text delegates to create_cache and rejects Gemini 4."""
        with patch("greycloud.cache.create_client"):
            cache = GreyCloudCache(argon_config)
            with pytest.raises(
                ValueError, match="automatic implicit durable caching in Vertex AI"
            ):
                cache.create_cache_from_text(
                    text="Paxos cluster node configuration details",
                    model="gemini-4-argon",
                )

    def test_min_tokens_by_model_excludes_gemini_4(self):
        """MIN_TOKENS_BY_MODEL table must not contain gemini-4 references."""
        assert "gemini-4-argon" not in MIN_TOKENS_BY_MODEL
        assert all("gemini-4" not in model_key for model_key in MIN_TOKENS_BY_MODEL)


# =====================================================================
# 5. Rate Limiter Thoughts Reconciliation Contract Tests
# =====================================================================


class TestGemini4RateLimiterReconciliationContract:
    """Verifies that actual reasoning tokens adjust rate limiter buckets."""

    @pytest.mark.asyncio
    async def test_rate_limiter_reconcile_usage_surplus(self):
        """reconcile_usage debits the positive delta from TPM bucket."""
        limiter = VertexRateLimiter(rpm=60, tpm=50000)
        mock_tpm = MagicMock()
        mock_tpm.acquire = AsyncMock()
        limiter._tpm_limiter = mock_tpm

        # Estimate: 100 prompt tokens; Actual: 350 (includes 250 thoughts)
        await limiter.reconcile_usage(estimated_tokens=100, actual_tokens=350)
        mock_tpm.acquire.assert_awaited_once_with(250)

    @pytest.mark.asyncio
    async def test_rate_limiter_reconcile_usage_no_surplus(self):
        """reconcile_usage does nothing when actual tokens <= estimated tokens."""
        limiter = VertexRateLimiter(rpm=60, tpm=50000)
        mock_tpm = MagicMock()
        mock_tpm.acquire = AsyncMock()
        limiter._tpm_limiter = mock_tpm

        await limiter.reconcile_usage(estimated_tokens=300, actual_tokens=250)
        mock_tpm.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limiter_reconcile_usage_clamps_to_tpm(self):
        """reconcile_usage clamps surplus debit to maximum TPM capacity."""
        limiter = VertexRateLimiter(rpm=60, tpm=2000)
        mock_tpm = MagicMock()
        mock_tpm.acquire = AsyncMock()
        limiter._tpm_limiter = mock_tpm

        # Surplus is 4900, but TPM capacity is 2000
        await limiter.reconcile_usage(estimated_tokens=100, actual_tokens=5000)
        mock_tpm.acquire.assert_awaited_once_with(2000)

    @pytest.mark.asyncio
    async def test_argon_unary_generation_reconciles_high_reasoning_tokens(
        self, argon_config, consensus_contents
    ):
        """Unary generate_content reconciles total token count with thinking tokens."""
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_response.text = "Consensus commit verified."
        mock_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=20,
            candidates_token_count=80,
            thoughts_token_count=500,
            total_token_count=600,
        )

        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            with patch.object(
                async_client.rate_limiter, "reconcile_usage", new_callable=AsyncMock
            ) as mock_reconcile:
                resp = await async_client.generate_content(
                    consensus_contents, model="gemini-4-argon"
                )
                assert resp.text == "Consensus commit verified."
                # Verify reconcile_usage was invoked with total_token_count=600
                mock_reconcile.assert_awaited_once()
                call_args = mock_reconcile.call_args[0]
                assert call_args[1] == 600

    @pytest.mark.asyncio
    async def test_argon_stream_generation_reconciles_on_completion(
        self, argon_config, consensus_contents
    ):
        """Stream reconciles total tokens upon stream completion."""
        chunk1 = _make_chunk("Analyzing distributed consensus quorum...", thought=True)
        chunk2 = _make_chunk(
            "Consensus quorum verified.", thought=False, total_tokens=850
        )

        with patch("greycloud.async_client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content_stream = MagicMock(
                side_effect=lambda **kw: _mock_stream_awaitable(chunk1, chunk2)
            )
            mock_create.return_value = mock_client

            async_client = GreyCloudAsyncClient(argon_config)
            with patch.object(
                async_client.rate_limiter, "reconcile_usage", new_callable=AsyncMock
            ) as mock_reconcile:
                async for _ in async_client.generate_content_stream(
                    consensus_contents, model="gemini-4-argon"
                ):
                    pass

                mock_reconcile.assert_awaited_once()
                call_args = mock_reconcile.call_args[0]
                assert call_args[1] == 850


# =====================================================================
# 6. _extract_total_tokens Robustness & Edge Cases Contract Tests
# =====================================================================


class TestExtractTotalTokensContract:
    """Verifies _extract_total_tokens handles dicts, objects, and fallbacks."""

    def test_extract_total_tokens_dict_direct_total(self):
        """Extracts total_token_count or totalTokenCount directly from dict."""
        assert (
            GreyCloudAsyncClient._extract_total_tokens({"total_token_count": 450})
            == 450
        )
        assert (
            GreyCloudAsyncClient._extract_total_tokens({"totalTokenCount": 620}) == 620
        )

    def test_extract_total_tokens_dict_fallback_sum_snake_case(self):
        """Sums prompt + candidates + thoughts in snake_case when total is missing."""
        meta = {
            "prompt_token_count": 50,
            "candidates_token_count": 150,
            "thoughts_token_count": 300,
        }
        assert GreyCloudAsyncClient._extract_total_tokens(meta) == 500

    def test_extract_total_tokens_dict_fallback_sum_camel_case(self):
        """Sums prompt + candidates + thoughts in camelCase when total is missing."""
        meta = {
            "promptTokenCount": 40,
            "candidatesTokenCount": 60,
            "thoughtsTokenCount": 200,
        }
        assert GreyCloudAsyncClient._extract_total_tokens(meta) == 300

    def test_extract_total_tokens_dict_partial_components(self):
        """Sums available components when thoughts are omitted."""
        meta = {
            "prompt_token_count": 80,
            "candidates_token_count": 120,
        }
        assert GreyCloudAsyncClient._extract_total_tokens(meta) == 200

    def test_extract_total_tokens_dict_invalid_total_triggers_fallback(self):
        """Non-integer total falls back to summing integer components."""
        meta = {
            "total_token_count": "invalid_string",
            "prompt_token_count": 25,
            "candidates_token_count": 75,
            "thoughts_token_count": 150,
        }
        assert GreyCloudAsyncClient._extract_total_tokens(meta) == 250

    def test_extract_total_tokens_dict_booleans_and_negatives_rejected(self):
        """Booleans (subclass of int) and negative totals are filtered out."""
        assert (
            GreyCloudAsyncClient._extract_total_tokens({"total_token_count": True})
            is None
        )
        assert (
            GreyCloudAsyncClient._extract_total_tokens({"total_token_count": False})
            is None
        )
        assert (
            GreyCloudAsyncClient._extract_total_tokens({"total_token_count": -50})
            is None
        )
        assert (
            GreyCloudAsyncClient._extract_total_tokens({"total_token_count": 0}) is None
        )
        assert (
            GreyCloudAsyncClient._extract_total_tokens(
                {"prompt_token_count": True, "candidates_token_count": 100}
            )
            is None
        )

    def test_extract_total_tokens_object_direct_total(self):
        """Extracts total_token_count directly from an object attribute."""

        class UsageObj:
            def __init__(self, total):
                self.total_token_count = total

        assert GreyCloudAsyncClient._extract_total_tokens(UsageObj(750)) == 750

    def test_extract_total_tokens_object_fallback_sum(self):
        """Sums attributes on object when total_token_count is absent."""

        class UsageObj:
            def __init__(self, p, c, t):
                self.prompt_token_count = p
                self.candidates_token_count = c
                self.thoughts_token_count = t

        assert GreyCloudAsyncClient._extract_total_tokens(UsageObj(30, 70, 200)) == 300

    def test_extract_total_tokens_none_and_unmocked_mock(self):
        """None or unconfigured MagicMock returns None safely without raising."""
        assert GreyCloudAsyncClient._extract_total_tokens(None) is None
        # Unconfigured MagicMock attributes fail isinstance(x, int)
        assert GreyCloudAsyncClient._extract_total_tokens(MagicMock()) is None
