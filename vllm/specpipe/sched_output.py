# SPDX-License-Identifier: Apache-2.0
"""Scheduler-output payloads specific to speculative pipeline parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request


@dataclass
class SpecpipeSchedulerOutput:
    """What SpecPipeScheduler hands to the workers for one iteration.

    Mirrors upstream ``SchedulerOutput`` and adds ``embeddings``: with
    specpipe enabled the first pipeline stage has no GPU embedding table,
    so the engine core looks the tokens up on the CPU (see
    handle_inputbatch) and ships the vectors alongside the schedule.
    """

    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: list[CachedRequestData]

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests.
    num_common_prefix_blocks: int

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of (req_id, encoder_input_index) tuples.
    # Used to free the encoder cache.
    free_encoder_input_ids: list[tuple[str, int]]

    # Dict of request ids to their index within the batch
    # for filling the next token bitmask
    structured_output_request_ids: dict[str, int]
    # the bitmask for the whole batch
    grammar_bitmask: Optional[npt.NDArray[np.int32]]

    # Input embeddings looked up on the CPU by the engine core; filled in
    # after scheduling (see EngineCore.step_with_specpipe).
    embeddings: Optional[torch.Tensor]


@dataclass
class SpecpipeSchedulerResumeHelper:
    """Token ids needed to rebuild requests resumed from preemption.

    A resumed request must re-send its full token history, which the
    CachedRequestData diff does not carry; the CPU embedding pass reads it
    from here (see handle_inputbatch.reorder_token_ids).
    """

    # req_id -> all token ids of the resumed request
    resumed_req_token_ids: dict[str, list[int]]

    @classmethod
    def from_request_list(
        cls,
        requests: list[Request],
    ) -> SpecpipeSchedulerResumeHelper:
        return cls(resumed_req_token_ids={
            request.request_id: request.all_token_ids
            for request in requests
        })
