# SPDX-License-Identifier: Apache-2.0
"""KV cache manager variant that never commits blocks to the prefix cache."""

from vllm.v1.core.kv_cache_manager import KVCacheManager


class SPKVCacheManager(KVCacheManager):
    """Upstream ``KVCacheManager`` minus prefix-cache population.

    Under Spexis, speculative tokens are written into KV blocks *before*
    verification; a rejected speculation rolls those tokens back. If
    ``allocate_slots`` committed such blocks to the prefix cache
    (``block_pool.cache_full_blocks``), the cache could serve unverified
    content to later requests. This subclass therefore skips the commit
    step — the single behavioral difference from upstream — while keeping
    block bookkeeping, ref-counting, and eviction identical.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_full_blocks has exactly one call site (allocate_slots);
        # neutralizing it on our own pool instance skips prefix-cache
        # population without copying the 120-line method body.
        self.block_pool.cache_full_blocks = self._skip_cache_full_blocks

    @staticmethod
    def _skip_cache_full_blocks(**kwargs) -> None:
        return None
