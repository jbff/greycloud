"""Tests for VertexRateLimiter"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from greycloud.rate_limiter import VertexRateLimiter


class TestVertexRateLimiterInit:
    """Tests for VertexRateLimiter initialization"""

    def test_default_init(self):
        """Default initialization uses baseline values"""
        limiter = VertexRateLimiter()
        assert limiter.rpm == 60
        assert limiter.tpm == 250_000
        assert limiter.max_concurrency == 10

    def test_custom_init(self):
        """Custom initialization overrides defaults"""
        limiter = VertexRateLimiter(rpm=30, tpm=100_000, max_concurrency=5)
        assert limiter.rpm == 30
        assert limiter.tpm == 100_000
        assert limiter.max_concurrency == 5


class TestVertexRateLimiterCallWithLimits:
    """Tests for call_with_limits method"""

    @pytest.mark.asyncio
    async def test_call_succeeds(self):
        """Successful API call returns response"""
        limiter = VertexRateLimiter(rpm=60, tpm=250_000, max_concurrency=10)
        mock_response = MagicMock()
        mock_coro = AsyncMock(return_value=mock_response)

        result = await limiter.call_with_limits(100, mock_coro())
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_call_propagates_exception(self):
        """Exceptions from API call propagate"""
        limiter = VertexRateLimiter(rpm=60, tpm=250_000, max_concurrency=10)
        mock_coro = AsyncMock(side_effect=RuntimeError("API error"))

        with pytest.raises(RuntimeError, match="API error"):
            await limiter.call_with_limits(100, mock_coro())

    @pytest.mark.asyncio
    async def test_concurrency_limited(self):
        """Concurrency is limited by semaphore"""
        limiter = VertexRateLimiter(rpm=1000, tpm=1_000_000, max_concurrency=2)
        max_concurrent = 0
        current_concurrent = 0

        async def track_concurrency():
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            current_concurrent -= 1
            return "done"

        tasks = [limiter.call_with_limits(10, track_concurrency()) for _ in range(5)]
        await asyncio.gather(*tasks)
        assert max_concurrent <= 2


class TestVertexRateLimiterEstimateTokens:
    """Tests for estimate_tokens static method"""

    def test_estimate_from_string(self):
        """Estimate tokens from string using len//4 heuristic"""
        text = "a" * 400  # 400 chars -> ~100 tokens
        assert VertexRateLimiter.estimate_tokens(text) == 100

    def test_estimate_empty_string(self):
        """Empty string returns 0"""
        assert VertexRateLimiter.estimate_tokens("") == 0

    @pytest.mark.asyncio
    async def test_tpm_acquisition_clamped_to_capacity(self, monkeypatch):
        """A request whose token estimate exceeds the TPM bucket capacity must
        still proceed (clamped to capacity, waiting for a fresh window at
        worst) instead of raising aiolimiter's 'Can't acquire more than the
        maximum capacity' ValueError — a single request cannot be split, and
        the server's real TPM quota is enforced by Vertex itself. Regression
        for BlackSheep's long-context prompts (injected knowledge block plus
        case history estimated above the 250K/min default bucket)."""
        limiter = VertexRateLimiter(rpm=60, tpm=250_000, max_concurrency=1)
        mock_coro = AsyncMock(return_value="done")

        # Estimate far above capacity: must not raise.
        result = await limiter.call_with_limits(500_000, mock_coro())
        assert result == "done"
        mock_coro.assert_awaited_once()

        # The bucket was fully consumed by the clamped acquisition: a second
        # request in the same window must wait for capacity to replenish
        # (not fail), proving the clamp actually acquired.
        # The clamped acquisition amount: patch the bucket class to record
        # the requested amount, asserting the request acquired exactly the
        # bucket capacity, not the raw estimate.
        import aiolimiter

        acquired = []
        real_acquire = aiolimiter.AsyncLimiter.acquire

        async def spy_acquire(self, amount=1):
            acquired.append(amount)
            await real_acquire(self, amount)

        monkeypatch.setattr(aiolimiter.AsyncLimiter, "acquire", spy_acquire)
        limiter2 = VertexRateLimiter(rpm=60, tpm=250_000, max_concurrency=1)
        await limiter2.call_with_limits(500_000, AsyncMock(return_value="x")())
        # acquired[0] is the RPM bucket's acquire(1); the TPM acquisition
        # (clamped to capacity, not the 500K estimate) comes last.
        assert acquired[-1] == 250_000
