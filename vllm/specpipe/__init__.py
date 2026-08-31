# SPDX-License-Identifier: Apache-2.0
"""Spexis: speculative pipeline parallelism on top of the vLLM V1 engine.

Modules:
    specpipe_scheduler          scheduler that overlaps speculation with
                                normal pipeline execution
    specpipe_kv_cache_manager   KV cache manager that keeps spec tokens out
                                of the prefix cache
    exitlayer / exit_models     per-stage early-exit (draft) models
    length_predictor            remaining-length prediction heads (sppp>=2.0)
    length_calc                 expected-generation-length / peak-KV math
    handle_inputbatch           CPU-side embedding batch bookkeeping
    shm                         engine-core <-> worker shared-memory channel
    bool_token                  Ray-DAG sentinel payload
"""
