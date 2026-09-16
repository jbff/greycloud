"""
Async rate limiting for Vertex AI / GenAI API calls.

Enforces RPM (requests per minute), TPM (tokens per minute), and concurrency
limits to stay within Vertex AI quotas.
"""

import asyncio
from aiolimiter import AsyncLimiter


class VertexRateLimiter:
    """
    Rate limiter for Vertex AI API calls.

    Enforces three limits simultaneously:
    - RPM: requests per minute (token bucket)
    - TPM: tokens per minute (token bucket with weighted acquisition)
    - Concurrency: max active requests (semaphore)

    Default baselines are for Tier 1/2 us-east4 quotas.
    """

    def __init__(self, rpm: int = 60, tpm: int = 250_000, max_concurrency: int = 10):
        self.rpm = rpm
        self.tpm = tpm
        self.max_concurrency = max_concurrency
        self._rpm_limiter = AsyncLimiter(rpm, 60)
        self._tpm_limiter = AsyncLimiter(tpm, 60)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def call_with_limits(self, prompt_tokens_est: int, api_call_coro):
        """
        Execute an API call while respecting rate limits.

        Args:
            prompt_tokens_est: Estimated token count for the request.
            api_call_coro: An awaitable API call (e.g. from client.aio.models).

        Returns:
            The API response.

        Raises:
            Whatever the underlying API call raises.
        """
        await self._rpm_limiter.acquire()
        # Clamp the TPM acquisition to the bucket capacity: a single request
        # cannot be split, so one whose estimate exceeds the per-minute budget
        # must still go through (it waits for a fresh window at worst) rather
        # than being categorically rejected. Without the clamp, any request
        # estimated above `tpm` — e.g. long-context prompts carrying a large
        # injected knowledge block — raises aiolimiter's
        # "Can't acquire more than the maximum capacity" ValueError on every
        # attempt and the client retries until the caller gives up. The
        # server's real TPM quota is enforced by Vertex itself.
        await self._tpm_limiter.acquire(min(prompt_tokens_est, self.tpm))
        async with self._semaphore:
            return await api_call_coro

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text using len//4 heuristic."""
        return len(text) // 4
