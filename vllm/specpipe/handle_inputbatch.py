# SPDX-License-Identifier: Apache-2.0
"""CPU-side embedding batch bookkeeping for the Spexis engine core.

With specpipe enabled, the first pipeline rank does not hold GPU embedding
weights (see the ``embed_tokens`` gate in models/llama.py, qwen2.py);
instead the engine-core process performs the embedding lookup on CPU and
ships the vectors to the worker inside the scheduler output. To do that,
this module mirrors the GPU-side persistent-batch bookkeeping
(``CPUInputBatch`` is a trimmed copy of gpu_input_batch's request-index
logic) so the CPU-assembled rows land in exactly the order
``gpu_model_runner`` expects.
"""
from typing import Optional
import torch
import itertools
import time

from vllm.specpipe.iteration_log import IterationRecord
from vllm.specpipe.sched_output import (SpecpipeSchedulerOutput,
                                        SpecpipeSchedulerResumeHelper)


class CPUInputBatch:
    def __init__(self, max_num_reqs):
        self.max_num_reqs = max_num_reqs
        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str,int] = {}
    
    def add_request(
        self,
        req_id: str,
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
        else:
            self._req_ids[req_index] = req_id
            
        self.req_id_to_index[req_id] = req_index

    
    def remove_request(self, req_id: str):
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None

        return req_index

    def condense(self, empty_req_indices: list[int]) -> None:
        num_reqs = self.num_reqs
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Swap the states.
            req_id = self._req_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs:]

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)
    

def update_states(input_batch: "CPUInputBatch",
                  scheduler_output: "SpecpipeSchedulerOutput") -> None:
    """Mirror the scheduler output onto the CPU persistent batch.

    Keeps ``input_batch``'s request->index mapping in sync with what the
    GPU-side input batch will compute from the same scheduler output, so
    that reorder_token_ids() can emit embedding rows in matching order.
    """
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    removed_req_indices: list[int] = []
    for req_id in scheduler_output.finished_req_ids:
        req_index = input_batch.remove_request(req_id)
        if req_index is not None:
            removed_req_indices.append(req_index)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = input_batch.req_id_to_index.keys()
    unscheduled_req_ids = cached_req_ids - scheduled_req_ids
    # NOTE: with pipeline-parallel micro-batching, consecutive steps
    # alternate between two in-flight batches, so this removal/re-add
    # churns every step. That is expected here — only the index mapping
    # matters for the CPU lookup, and it must track the GPU batch exactly.
    for req_id in unscheduled_req_ids:
        req_index = input_batch.remove_request(req_id)
        assert req_index is not None
        removed_req_indices.append(req_index)

    req_ids_to_add: list[str] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        req_ids_to_add.append(req_id)

    # Requests already present in the batch need no CPU-side state update;
    # we only re-add the ones that dropped out (preempted or unscheduled).
    for req_data in scheduler_output.scheduled_cached_reqs:
        req_id = req_data.req_id

        req_index = input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.
            req_ids_to_add.append(req_id)
            continue

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    removed_req_indices = sorted(removed_req_indices, reverse=True)
    for req_id in req_ids_to_add:
        if removed_req_indices:
            # Fill the empty index.
            req_index = removed_req_indices.pop()
        else:
            # Append to the end.
            req_index = None
        input_batch.add_request(req_id, req_index)

    # Condense the batched states if there are empty indices.
    if removed_req_indices:
        input_batch.condense(removed_req_indices)


def reorder_token_ids(
    input_batch: "CPUInputBatch",
    scheduler_output: "SpecpipeSchedulerOutput",
    resume_helper: "SpecpipeSchedulerResumeHelper"
) -> torch.Tensor:
    """Collect this step's token ids in the batch's request-index order."""
    ordered_tokens_by_req = [None] * input_batch.num_reqs
    
    for req_data in scheduler_output.scheduled_new_reqs:
        req_index = input_batch.req_id_to_index.get(req_data.req_id)
        if req_index is not None:
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_data.req_id, 0)
            ordered_tokens_by_req[req_index] = req_data.prompt_token_ids[:num_scheduled]

    for req_data in scheduler_output.scheduled_cached_reqs:
        req_index = input_batch.req_id_to_index.get(req_data.req_id)
        assert req_index is not None, "reorder token no req_index bug"

        if req_data.resumed_from_preemption:
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_data.req_id, 0)
            ordered_tokens_by_req[req_index] = resume_helper.resumed_req_token_ids[req_data.req_id][:num_scheduled]
        else:
            ordered_tokens_by_req[req_index] = req_data.new_token_ids

    
    flat_token_ids = itertools.chain.from_iterable(
        tokens for tokens in ordered_tokens_by_req if tokens is not None
    )
    
    return torch.tensor(list(flat_token_ids), dtype=torch.long)


def total_embedding_process(
    scheduler_output: "SpecpipeSchedulerOutput",
    input_batch: "CPUInputBatch",
    embed_tokens: torch.Tensor,
    rec: "IterationRecord",
    resume_helper: "SpecpipeSchedulerResumeHelper",
) -> None:
    """Assemble this step's input embeddings on the CPU.

    Syncs the persistent batch, gathers token ids in batch order, looks
    them up in the pinned ``embed_tokens`` weight, and attaches the result
    to ``scheduler_output.embeddings`` for shipment to the first-stage
    worker. Per-stage wall times are recorded into ``rec``.
    """
    torch.cuda.nvtx.range_push(msg="cpu embedding lookup")
    first = time.perf_counter()
    # reorder requests & tokens to allign with gpu_model_runner
    update_states(input_batch, scheduler_output)
    # make token id tensor (use self.input_batch's req_index order)
    second = time.perf_counter()
    token_ids = reorder_token_ids(input_batch, scheduler_output, resume_helper)
    third = time.perf_counter()
    input_embeds = embed_tokens[token_ids]
    fourth = time.perf_counter()
    scheduler_output.embeddings = input_embeds
    torch.cuda.nvtx.range_pop()

    rec.set_cpu_embed_times(update_states=second - first,
                            reorder=third - second,
                            lookup=fourth - third)
