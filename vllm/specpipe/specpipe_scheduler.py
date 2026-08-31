# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from typing import Optional, Union

from vllm.config import (CacheConfig, LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, SpecPipeConfig)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.specpipe.length_calc import (ExpGenLenCache, LenProbCache,
                                       calculate_peak_mem)
from vllm.specpipe.sched_output import (SpecpipeSchedulerOutput,
                                        SpecpipeSchedulerResumeHelper)
from vllm.specpipe.specpipe_kv_cache_manager import SPKVCacheManager
from vllm.specpipe.stop_check import (check_stop_from_spec,
                                      check_stop_from_validation)
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)



class SpecPipeScheduler(SchedulerInterface):
    """Scheduler for speculative pipeline parallelism (Spexis).

    Differences from the upstream V1 scheduler:
    - Two-phase updates per pipeline iteration: update_from_spec_output
      ingests exit-layer (draft) tokens; update_from_true_output verifies
      them against the target model and rolls back mismatches.
    - Confidence-based spec drop (sppp >= 1.5): per-iteration bucket sort
      by draft confidence; only the top (1 - drop_ratio) fraction keeps
      speculating, plus optional short-request acceleration (sppp 2.5).
    - Lookahead prefill admission (sppp >= 2.0): projected peak KV usage
      (see length_calc) delays new prefills that would overflow memory.

    schedule() returns (SpecpipeSchedulerOutput, resume_helper,
    max_dropped_confidence, len_estimate_time).
    """

    def __init__(self,
                 scheduler_config: SchedulerConfig,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 lora_config: Optional[LoRAConfig],
                 kv_cache_config: KVCacheConfig,
                 structured_output_manager: StructuredOutputManager,
                 speculative_config: SpeculativeConfig = None,
                 mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
                 include_finished_set: bool = False,
                 log_stats: bool = False,
                 parallel_config: ParallelConfig = None,
                 specpipe_config: SpecPipeConfig = None) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.kv_cache_config = kv_cache_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        self.sppp_len_pred = specpipe_config.sppp_len_pred
        self.sppp_ver = specpipe_config.sppp_ver
        # Absolute confidence cutoff (0-1) for spec dropping; unused when
        # drop_ratio drives relative dropping.
        self.confidence_threshold = specpipe_config.confidence_threshold
        # Fraction (0-1) of the batch dropped from speculation each
        # iteration, lowest confidence first.
        self.drop_ratio = specpipe_config.drop_ratio
        self.specpipe_pp_size = parallel_config.pipeline_parallel_size

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.include_finished_set = include_finished_set

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        # Create the KV cache manager.
        self.kv_cache_manager = SPKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            log_stats=self.log_stats)
        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()

        # Lookahead-scheduling knobs; values come from SpecPipeConfig
        # (its defaults reproduce the paper setup).
        self.mem_esti_bar = specpipe_config.mem_esti_bar
        self.mem_estimate_margin = specpipe_config.mem_estimate_margin
        self.average_spec_acc = specpipe_config.average_spec_acc
        self.max_iter = specpipe_config.max_estimate_iter
        self.iter_list = list(
            range(specpipe_config.estimate_iter_stride, self.max_iter + 1,
                  specpipe_config.estimate_iter_stride))
        self.expect_length = specpipe_config.expect_length
        self.lenprob_threshold = specpipe_config.lenprob_threshold
        self.lenesti_prec = specpipe_config.lenesti_prec
        self.num_conf_buckets = specpipe_config.confidence_buckets
        self.prefill_reserve_blocks = specpipe_config.prefill_reserve_blocks
        if self.sppp_len_pred:
            self.expect_gen_len_cache = ExpGenLenCache(
                self.max_iter, self.specpipe_pp_size, self.average_spec_acc)
            self.len_prob_cache = LenProbCache(
                self.max_iter, self.specpipe_pp_size, self.average_spec_acc,
                self.expect_length)
        else:
            self.expect_gen_len_cache = None
            self.len_prob_cache = None

        # In-flight (scheduled at least once, not finished) requests.
        self.running_scheduled: list[Request] = []

        # Short-request extra acceleration (sppp 2.5) knobs.
        self.enable_add_acc = self.sppp_ver == "2.5"
        self.max_add_reqs = specpipe_config.max_add_reqs
        self.add_acc_lenidx_thre = specpipe_config.add_acc_lenidx_thre
        self.add_acc_conf_thre = specpipe_config.add_acc_conf_thre

        logger.info("SpecPipeScheduler: sppp_ver=%s, len_pred=%s, add_acc=%s",
                    self.sppp_ver, self.sppp_len_pred, self.enable_add_acc)
        
        # Requests marked for preemption whose in-flight iterations have not
        # drained yet (see _preempt_request).
        self.to_be_preempted: list[Request] = []

        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> CachedRequestData
        self._cached_reqs_data: dict[str, CachedRequestData] = {}

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        self.num_lookahead_tokens = 0
        if speculative_config and speculative_config.method == "eagle":
            self.num_lookahead_tokens = \
                speculative_config.num_speculative_tokens

    def schedule(self):
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []

        scheduled_running_flushed_reqs: list[Request] = []
        scheduled_running_specpp_reqs: list[Request] = []

        preempted_reqs: list[Request] = []

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # for sort & cut high score requests
        num_guaranteed_reqs = 0
        # Zero, try to preempt needed requests
        new_to_be_preempted: list[Request] = []
        
        req_idx = 0
        while req_idx < len(self.to_be_preempted):
            preempt_req = self.to_be_preempted[req_idx]
            if not self._preempt_request(preempt_req, scheduled_timestamp):
                new_to_be_preempted.append(preempt_req)
            preempted_reqs.append(preempt_req)
            req_idx += 1
        self.to_be_preempted = new_to_be_preempted

        drop_ratio = self.drop_ratio
        batch_size = len(self.running_scheduled)
        effective_pass_num = batch_size - int(batch_size * drop_ratio)
        num_real_requests = 0  
        num_buckets = self.num_conf_buckets
        bucket_width = 1.0 / num_buckets
        # Histogram over confidence: [0, w), [w, 2w), ..., [1 - w, 1.0].
        confidence_bucket_list = [[] for _ in range(num_buckets)]
        
        # bucket sorting (by spec confidence)
        req_idx = 0
        while req_idx < len(self.running_scheduled):
            request = self.running_scheduled[req_idx]
            if request.should_preempt:
                request.confidence_passed = False
                req_idx += 1
                continue
            # Check Real Token only requests (from spec revision or from drop or chunked prefill)
            if self._has_pending_real_token(request):
                num_real_requests += 1
                request.confidence_passed = False
                req_idx += 1
                continue
            elif request.confidence_dropped or request.stop_dropped:
                request.confidence_passed = False
                req_idx += 1
                continue
            else:
                request.confidence_passed = False
                confidence_score = request.spec_confidence()
                bucket_idx = min(int(confidence_score / bucket_width), num_buckets - 1)
                if bucket_idx < 0 or bucket_idx >= num_buckets:
                    raise ValueError(f"Invalid confidence score: {confidence_score} from request {request.request_id}")
                confidence_bucket_list[bucket_idx].append(request)
                req_idx += 1
                continue
        
        # Choose Drop Requests
        max_dropped_confidence = 0
        effective_pass_num = max(effective_pass_num - num_real_requests, 0)
        tmp_idx = num_buckets - 1
        tmp_req_idx = 0
        while effective_pass_num > 0 and tmp_idx >= 0:
            if len(confidence_bucket_list[tmp_idx]) <= effective_pass_num:
                for request in confidence_bucket_list[tmp_idx]:
                    request.confidence_passed = True
                effective_pass_num -= len(confidence_bucket_list[tmp_idx])
                max_dropped_confidence = tmp_idx * bucket_width
            else:
                while effective_pass_num > 0 and tmp_req_idx < len(confidence_bucket_list[tmp_idx]):
                    request = confidence_bucket_list[tmp_idx][tmp_req_idx]
                    request.confidence_passed = True
                    effective_pass_num -= 1
                    tmp_req_idx += 1  
                max_dropped_confidence = confidence_bucket_list[tmp_idx][tmp_req_idx].spec_confidence() 
            tmp_idx -= 1
        
        # scheme2.5 additional acceleration
        if self.enable_add_acc:
            max_add_acc_req = self.max_add_reqs
            if tmp_req_idx != 0:
                while tmp_req_idx < len(confidence_bucket_list[tmp_idx]):
                    request = confidence_bucket_list[tmp_idx][tmp_req_idx]
                    if (request.expect_remain_len_idx(self.lenprob_threshold) == self.add_acc_lenidx_thre) and (request.spec_confidence()>self.add_acc_conf_thre) and (max_add_acc_req > 0):
                        request.confidence_passed = True
                        max_add_acc_req -= 1
                    
                    # break while (one bucket)
                    if max_add_acc_req==0:
                            break
                    tmp_req_idx += 1
                tmp_idx -= 1
            
            if max_add_acc_req != 0:
                while tmp_idx >=0:
                    for request in confidence_bucket_list[tmp_idx]:
                        if (request.expect_remain_len_idx(self.lenprob_threshold) == self.add_acc_lenidx_thre) and (request.spec_confidence()>self.add_acc_conf_thre) and (max_add_acc_req > 0):
                            request.confidence_passed = True
                            max_add_acc_req -= 1
                        # break for (one bucket)
                        if max_add_acc_req==0:
                            break
                    # break while (whole bucket)
                    if max_add_acc_req==0:
                            break
                    
                    tmp_idx -= 1

        num_dropped = 0
        dropped_reqids = []
        
        # if no prefill available -> threshold = -1000, if available -> threshold = pre-fixed value

        # First, schedule flush request and sort by confidence
        req_index = 0
        while req_index < len(self.running_scheduled) and token_budget > 0:
            request = self.running_scheduled[req_index]

            # 0) Do not scedule preempt requests
            if request.should_preempt:
                req_index += 1
                continue
            
            # 1-0) Schedule target model generated token (from spec revision or from drop or chunked prefill)
            if self._has_pending_real_token(request):
                # Flushed -> must add to batch!!

                request.drop_end = False

                if request.num_computed_tokens < request.num_prompt_tokens:
                    num_new_tokens = (request.num_tokens_with_spec -
                                    request.num_computed_tokens) # for chunked prefill
                else:
                    num_new_tokens = 1  # exactly the corrected token from verification
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # KV blocks for the spec tokens were already allocated
                # during the spec step; the manager extends them here.
                new_blocks = self._allocate_with_preemption(
                    request, num_new_tokens, scheduled_timestamp,
                    preempted_reqs)
                if new_blocks is None:
                    break

                # Schedule the request.
                scheduled_running_flushed_reqs.append(request)
                request.num_scheduled += 1

                self.scheduled_req_ids.add(request.request_id)
                if request.use_structured_output:
                    # PERF: in case of chunked prefill,
                    # request might not include any new tokens.
                    # Therefore, we might introduce some additional
                    # cycle to fill in the bitmask, which could be a big no-op.
                    structured_output_request_ids[
                        request.request_id] = req_index

                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                num_guaranteed_reqs += 1
                req_index += 1
            # 1-1) Pass Dropped requests
            elif request.confidence_dropped or request.stop_dropped:
                req_index += 1
                continue
            # 1-2) Schedule high prob reqs and Drop low prob reqs
            else:  # True or None
                # 1-2.a) Schedule high prob reqs
                if request.confidence_passed:
                    num_new_tokens = (request.num_tokens_with_spec -
                                    request.num_computed_tokens)
                    if (0 < self.scheduler_config.long_prefill_token_threshold <
                            num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)
                    num_new_tokens = min(num_new_tokens, token_budget)
                    
                    assert num_new_tokens > 0

                    new_blocks = self._allocate_with_preemption(
                        request, num_new_tokens, scheduled_timestamp,
                        preempted_reqs)
                    if new_blocks is None:
                        break

                    # Schedule the request.
                    scheduled_running_specpp_reqs.append(request)
                    request.num_scheduled += 1

                    self.scheduled_req_ids.add(request.request_id)

                    if request.use_structured_output:
                        # PERF: in case of chunked prefill,
                        # request might not include any new tokens.
                        # Therefore, we might introduce some additional
                        # cycle to fill in the bitmask, which could be a big no-op.
                        structured_output_request_ids[request.request_id] = req_index

                    req_to_new_block_ids[request.request_id] = [
                        b.block_id for b in new_blocks
                    ]
                    num_scheduled_tokens[request.request_id] = num_new_tokens
                    token_budget -= num_new_tokens
                    req_index += 1  
                # 1-2.b) Drop low prob reqs
                else:

                    request.confidence_dropped = True
                    request.drop_len =  request.specpipe_len

                    req_index += 1
                    pass

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()

        
        should_estimate_length = False
        delay_prefill = False
        len_estimate_time=0
        # scheme 2.0
        if self.sppp_len_pred:
            len_estimate_start_time = time.perf_counter()
            if not preempted_reqs:
                if len(self.waiting) > 0 and token_budget > 0 and self.kv_cache_manager.usage > self.mem_esti_bar:
                    # length prediction for waiting requests
                    should_estimate_length = True
                    # Reserve headroom for per-request partially filled
                    # blocks (fragmentation).
                    mem_in_tokens = (
                        (self.kv_cache_manager.block_pool.num_gpu_blocks -
                         self.prefill_reserve_blocks) * self.block_size)

                    mem_usage_list, peak_mem, peak_iter = calculate_peak_mem(
                        exp_gen_len_cache=self.expect_gen_len_cache,
                        len_prob_cache=self.len_prob_cache,
                        iter_list=self.iter_list, # hard coded max iter for length prediction
                        current_batch=self.running_scheduled,
                        length_thresholds=self.lenprob_threshold, # hard coded length thresholds for length prediction
                        expect_precision=self.lenesti_prec, #self.scheduler_config.length_predictor_precisions,
                    )

                    if peak_mem > self.mem_estimate_margin * mem_in_tokens:
                        delay_prefill = True
            len_estimate_end_time = time.perf_counter()
            len_estimate_time = len_estimate_end_time - len_estimate_start_time

        #Fourth, schedule the WAITING requests.
        if not preempted_reqs and not delay_prefill: # for Strict FCFS: and len(self.to_be_preempted) == 0
            while self.waiting and token_budget > 0:
                if len(self.running_scheduled) == self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                if should_estimate_length:
                    for i, iter_val in enumerate(self.iter_list, start=1):
                        mem_usage_list[i-1] += (request.num_tokens + self.expect_gen_len_cache.get(iter_val))
                        if mem_usage_list[i-1] > peak_mem:
                            peak_mem = mem_usage_list[i-1]
                            peak_iter = iter_val
                        
                        if mem_usage_list[i-1] > self.mem_estimate_margin * mem_in_tokens:
                            delay_prefill = True
                            break
                    # Do Not Schedule Prefill if it may cause OOM based on length prediction
                    if delay_prefill:
                        break

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Get already-cached tokens. <- related to prefix caching
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(request)
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)

                assert num_new_tokens > 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_blocks)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break
                
                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running_scheduled.append(request)
                request.num_scheduled += 1
                self.scheduled_req_ids.add(request.request_id)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running_scheduled) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_flushed_reqs) + len(scheduled_running_specpp_reqs) <= len(self.running_scheduled)), \
                    f"{len(scheduled_new_reqs)}+{len(scheduled_resumed_reqs)}+{len(scheduled_running_flushed_reqs)}+{len(scheduled_running_specpp_reqs)} > {len(self.running_scheduled)}"

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running_scheduled),
        )
        if grammar_bitmask is not None:
            raise NotImplementedError

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]

        # NOTE: cached-req groups are concatenated in (resumed, flushed,
        # spec) order; the worker-side reorder relies on this order.
        running_flushed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_flushed_reqs
        ]

        running_specpp_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_specpp_reqs
        ]

        scheduler_output = SpecpipeSchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data +
            running_flushed_reqs_data + running_specpp_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=0,  #num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=None,  #grammar_bitmask,
            embeddings=None, # will be updated later in core.py
        )

        resume_helper = SpecpipeSchedulerResumeHelper.from_request_list(scheduled_resumed_reqs)

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        return scheduler_output, resume_helper, max_dropped_confidence, len_estimate_time

    def _allocate_with_preemption(
            self, request: Request, num_new_tokens: int,
            scheduled_timestamp: float, preempted_reqs: list[Request]):
        """Allocate KV slots, preempting newest requests on failure.

        Returns the new blocks, or None if `request` itself had to be
        preempted (nothing left to evict). NOTE: eviction pops the tail of
        running_scheduled (newest first), which differs from upstream's
        lowest-priority preemption.
        """
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens)
            if new_blocks is not None:
                return new_blocks
            preempted_req = self.running_scheduled.pop()
            logger.debug("preempting %s to free KV blocks",
                         preempted_req.request_id)
            self.kv_cache_manager.free(preempted_req)
            preempted_req.status = RequestStatus.PREEMPTED
            if self.log_stats:
                preempted_req.record_event(EngineCoreEventType.PREEMPTED,
                                           scheduled_timestamp)
            if not self._preempt_request(preempted_req, scheduled_timestamp):
                self.to_be_preempted.append(preempted_req)
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                return None

    @staticmethod
    def _has_pending_real_token(request: Request) -> bool:
        """True if this request's next step must run a real (target-model)
        token: a just-corrected spec miss, a drained drop, or an
        unfinished (chunked) prefill."""
        return ((request.spec_result == False and request.just_fixed == True)
                or (request.drop_end == True)
                or (request.num_computed_tokens < request.num_prompt_tokens))

    def update_from_output(
        self,
        scheduler_output: SpecpipeSchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        raise NotImplementedError

    def _make_cached_request_data(
        self,
        request: Request,
        num_scheduled_tokens: int,
        num_scheduled_spec_tokens: int,
        new_block_ids: list[int],
        resumed_from_preemption: bool,
    ) -> CachedRequestData:
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        num_computed_tokens = request.num_computed_tokens
        num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens
        new_token_ids = request.all_token_ids[
            num_computed_tokens:num_computed_tokens + num_regular_tokens]
        req_data = self._cached_reqs_data.get(request.request_id)
        is_cached = req_data is not None

        # Resolve which correction (if any) this step carries to the worker.
        if request.postpreempt_spec_result is False:
            # Preemption decided right after a prefill iteration: replay the
            # correction that was recorded at preemption time.
            new_token_ids = [
                request.output_token_ids[request.postpreempt_fixed_idx]
            ]
            spec_result = False
            fixed_idx = request.postpreempt_fixed_idx
            request.postpreempt_spec_result = None
            request.postpreempt_fixed_idx = None
        elif request.spec_result is False and (request.just_fixed
                                               or not is_cached):
            # Speculation was rejected: send exactly the corrected token.
            # NOTE: a cached entry additionally requires just_fixed, so a
            # stale rejection is not resent on later steps.
            new_token_ids = [request.output_token_ids[request.fixed_idx]]
            spec_result = False
            fixed_idx = request.fixed_idx
        else:
            spec_result = True
            fixed_idx = -1

        if is_cached:
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_token_ids = new_token_ids
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            req_data = CachedRequestData.from_request(request,
                                                      resumed_from_preemption,
                                                      new_token_ids,
                                                      new_block_ids)
            self._cached_reqs_data[request.request_id] = req_data
        req_data.spec_result = spec_result
        req_data.fixed_idx = fixed_idx

        return req_data

    def update_from_spec_output(
        self,
        scheduler_output: SpecpipeSchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_confidence_list = model_runner_output.spec_confidence
        predicted_len_prob_list = model_runner_output.length_probs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        outputs: list[EngineCoreOutput] = []
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(self.running_scheduled) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        # NOTE: iterates the full running list; unscheduled entries skip
        # out early on num_tokens_scheduled == 0.
        for request in self.running_scheduled:
            req_id = request.request_id

            # micro-batch & preempted requests are in the running but not scheduled
            num_tokens_scheduled = num_scheduled_tokens.get(req_id,0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]
            generated_spec_confidence = spec_confidence_list[req_index]
            predicted_len_prob = (
                predicted_len_prob_list[req_index]
                if predicted_len_prob_list is not None
                else None
            )
            # do not check stop in speculation
            new_logprobs = None
            new_token_ids = generated_token_ids

            # Append generated tokens and check for stop. Note that if
            # a request is still being prefilled, we expect the model runner
            # to return empty token ids for the request.
            for num_new, output_token_id in enumerate(new_token_ids, 0):
                request.append_output_token_ids(output_token_id, generated_spec_confidence[num_new], predicted_len_prob)
                request.specpipe_len += 1
                stopped = check_stop_from_spec(request, self.max_model_len)
                if stopped:
                    request.stop_dropped = True
                    request.drop_len = request.specpipe_len

            if new_token_ids and request.use_structured_output:
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events()))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(spec_decoding_stats),
        )

        if self.include_finished_set:
            #TODO currently sending duplicates here, improve this
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids)

        return engine_core_outputs

    def update_from_true_output(
        self,
        scheduler_output: SpecpipeSchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        '''
        Do verify & fix
        '''
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        outputs: list[EngineCoreOutput] = []
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # Finished request ids to prune from the tracking lists below.
        free_reqs_scheduled: list[str] = []

        # Tallies of verify result codes 0..4:
        # (correct, wrong, flush_norm, flush_drop, final_drop)
        result_counts = [0, 0, 0, 0, 0]

        self._verify_and_collect(self.running_scheduled,
                                 num_scheduled_tokens, model_runner_output,
                                 outputs, free_reqs_scheduled, result_counts)
        self._verify_and_collect(self.to_be_preempted, num_scheduled_tokens,
                                 model_runner_output, outputs,
                                 free_reqs_scheduled, result_counts)
        (num_correct, num_wrong, num_flush_normal, num_flush_dropped,
         num_drop_final) = result_counts

        # Remove finished requests from the tracking lists.
        for target_id in free_reqs_scheduled:
            self.scheduled_req_ids.discard(target_id)
            # Linear scan; fine at experiment batch sizes.
            for i, req in enumerate(self.running_scheduled):
                if req.request_id == target_id:
                    del self.running_scheduled[i]
                    break

            for i, req in enumerate(self.to_be_preempted):
                if req.request_id == target_id:
                    del self.to_be_preempted[i]
                    break

        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(spec_decoding_stats),
        )
        if self.include_finished_set:
            #TODO currently sending duplicates here, improve this
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids)

        return engine_core_outputs , num_correct, num_wrong, num_flush_normal, num_flush_dropped, num_drop_final

    def _verify_and_collect(
        self,
        requests: list[Request],
        num_scheduled_tokens: dict[str, int],
        model_runner_output: ModelRunnerOutput,
        outputs: list[EngineCoreOutput],
        free_reqs_scheduled: list[str],
        result_counts: list[int],
    ) -> None:
        """Verify spec tokens for one request list (update_from_true_output).

        Appends user-visible outputs to `outputs`, finished request ids to
        `free_reqs_scheduled`, and tallies verify_output_token_ids result
        codes (0=correct, 1=wrong, 2=flush_norm, 3=flush_drop,
        4=final_drop) into `result_counts`.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        for request in requests:
            req_id = request.request_id
            if num_scheduled_tokens.get(req_id, 0) == 0:
                # The request was not verified in this step.
                continue
            request.num_scheduled -= 1
            req_index = model_runner_output.req_id_to_index[req_id]
            new_token_ids = sampled_token_ids[req_index]
            new_logprobs = None

            # A just-corrected request already emitted its token through
            # the flush path; suppress duplicate user output.
            send_to_user = request.spec_result is not False

            for num_new, output_token_id in enumerate(new_token_ids, 1):
                spec_result = request.verify_output_token_ids(output_token_id)
                assert 0 <= spec_result <= 4, (
                    f"unexpected verify result {spec_result}")
                result_counts[spec_result] += 1
                # Must run before building the EngineCoreOutput.
                stopped = check_stop_from_validation(request,
                                                     self.max_model_len)
                if stopped:
                    self._free_request(request)
                    free_reqs_scheduled.append(request.request_id)
                    del new_token_ids[num_new:]
                    break

            if not send_to_user:
                continue

            if request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and request.use_structured_output:
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events()))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                # NOTE: the request is not removed from running_scheduled
                # here; benchmark workloads never abort mid-run, so this
                # external-abort path is effectively unused today.
                self.scheduled_req_ids.discard(request.request_id)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)
    
    def _preempt_request(self, request: Request, timestamp) -> bool:
        """Mark `request` for preemption and finish it if possible.

        Returns True when the request was actually reset and moved back to
        the waiting queue; False when it still has in-flight iterations
        (num_scheduled > 0) and must wait in to_be_preempted until they
        drain.
        """
        request.should_preempt = True

        # Apply Spec Fix should be done right after preemption decision (not after real preemption)
        if request.postpreempt_spec_result is not False:
            request.postpreempt_spec_result = not(request.just_fixed and request.spec_result is False)
            request.postpreempt_fixed_idx = request.fixed_idx if request.postpreempt_spec_result is False else None

        if request.num_scheduled > 0:
            return False
        
        # re-initialize request
        request.specpipe_len = 0
        request.spec_result = None
        request.fixed_idx = None
        request.flush_len = None
        request.just_fixed = None
        
        request.drop_len = 0
        request.confidence_dropped = False
        request.stop_dropped = False
        request.confidence_passed = False
        request.predicted_len_prob = []

        request.should_preempt = False
        request.num_scheduled = 0

        request.num_computed_tokens = 0
        self.scheduled_req_ids.discard(request.request_id)
        self.waiting.appendleft(request)
        
        return True

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(
            self.running_scheduled) + len(self.to_be_preempted)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def get_num_unscheduled_requests(self) -> int:
        """Number of requests that are not being processed by the executor."""
        return self.get_num_unfinished_requests() - len(self.scheduled_req_ids)

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        return SchedulerStats(
            num_running_reqs=len(self.running_scheduled),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=self.kv_cache_manager.make_prefix_cache_stats(),
            spec_decoding_stats=spec_decoding_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats()
        spec_decoding_stats.observe(num_draft_tokens=num_draft_tokens,
                                    num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats
