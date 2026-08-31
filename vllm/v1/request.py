# SPDX-License-Identifier: Apache-2.0

import enum
from typing import TYPE_CHECKING, Optional, Union

from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest


class Request:

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: list[int],
        multi_modal_inputs: Optional[list[MultiModalKwargs]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request

        self.status = (RequestStatus.WAITING_FOR_FSM
                       if sampling_params.guided_decoding is not None else
                       RequestStatus.WAITING)
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0

        # [Spexis] Speculation bookkeeping. specpipe_len counts the tokens
        # appended speculatively but not yet verified; spec_result/fixed_idx
        # carry the outcome of the last verification.
        self.specpipe_len = 0
        self.spec_result = None
        self.fixed_idx = None
        self.flush_len = None
        self.just_fixed = None

        # specpipe confidence scheduling related
        self.spec_confidence_list: list[float] = []
        self.confidence_dropped = False
        self.stop_dropped = False
        self.confidence_passed = False
        self.drop_len = 0
        self.drop_end = False

        # specpipe special free/premempt logic
        self.should_preempt = False
        self.num_scheduled = 0

        # specpipe post-preemption info
        self.postpreempt_spec_result = None
        self.postpreempt_fixed_idx = None

        # [Spexis] Per-request speculation counters (diagnostics only).
        self.total_spec = 0
        self.spec_success = 0
        self.spec_fail = 0

        # specpipe length prediction related
        self.predicted_len_prob = []

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: list[str] = multi_modal_hashes or []
        self.num_encoder_inputs = len(self.mm_inputs)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

        # Read-only views
        # Prevent directly appending to the these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(request.mm_inputs, MultiModalKwargs), (
                "mm_inputs was not updated in EngineCore.add_request")

        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params),
        )

    def spec_confidence(self) -> float:
        """Joint confidence of the tokens still awaiting verification."""
        spec_confidence = 1.0
        for i in range(1, self.specpipe_len + 1):
            spec_confidence *= self.spec_confidence_list[-i]

        return spec_confidence

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
        spec_confidence: Optional[float] = None,
        predicted_len_prob: Optional[list[float]] = None,
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        # [Spexis] Only the speculative append (update_from_spec_output)
        # supplies a confidence, so this per-token state stays empty on the
        # vanilla path. It must stay index-aligned with the output tokens:
        # spec_confidence() reads it back by negative index.
        if spec_confidence is not None:
            self.spec_confidence_list.append(spec_confidence)
            self.predicted_len_prob = predicted_len_prob

    def verify_output_token_ids(
        self,
        token_ids: int,
    ) -> int:
        '''
        Verify specpipe spec token. Only supports greedy decoding for now.
        returns bool (whether spec was right), last_idx for _output_token_ids (0 for True)
        '''
        if isinstance(token_ids, int):
            # Handle Drop # times
            if self.confidence_dropped == True or self.stop_dropped == True:
                # handle previous flushed output
                # maliscious case: flush + drop ...
                if self.spec_result == False:
                    self.just_fixed = False
                    self.flush_len -= 1
                    if self.flush_len == 0:
                        self.spec_result = None
                    return 3
                
                self.drop_len -= 1
                if self.drop_len == 0:
                    self.stop_dropped = False
                    self.confidence_dropped = False
                    if self.spec_result == True or self.spec_result == None:
                        self._output_token_ids[-1:] = [token_ids]
                        self._all_token_ids[-1:] = [token_ids]
                        self.spec_confidence_list[-1:] = [1] # set absolute confidence for target generated fixed tokens
                        self.drop_end = True

                    self.spec_result = None
                    self.flush_len = None
                    self.specpipe_len = 0
                    self.fixed_idx = 0
                    return 4
                
                # print("Dropping But Verified as normal")
                
            # Validate Spec Result (Since token id is valid)
            if self.spec_result == True or self.spec_result == None:
                # until now, support greedy decoding only!
                if self._output_token_ids[-self.specpipe_len] == token_ids:
                    self.specpipe_len -= 1

                    self.spec_result = True
                    self.fixed_idx = 0
                    self.spec_success += 1
                    self.total_spec += 1
                    return 0
                else:
                    self.num_computed_tokens -= (
                        self.specpipe_len - 1
                    )  # at least one token was generate
                    false_spec_place = len(
                        self._output_token_ids) - self.specpipe_len

                    self._output_token_ids[-self.specpipe_len:] = [token_ids]
                    self._all_token_ids[-self.specpipe_len:] = [token_ids]

                    self.spec_confidence_list[-self.specpipe_len:] = [1] # set absolute confidence for target generated fixed tokens

                    self.flush_len = self.specpipe_len - 1
                    self.specpipe_len = 0
                    self.just_fixed = True
                    
                    # Reset drop flags because previous speculation was wrong -> drop was invalid
                    self.confidence_dropped = False
                    self.stop_dropped = False
                    self.drop_len = 0

                    self.spec_result = False
                    self.fixed_idx = false_spec_place
                    # print(f"spec false request: name[{self.request_id}], idx[{self.fixed_idx}]")
                    self.spec_fail += 1
                    self.total_spec += 1
                    return 1

            # Flush token id & Handle Flush # times (Since token id is not valid)
            else:
                self.just_fixed = False
                self.flush_len -= 1
                if self.flush_len == 0:
                    self.spec_result = None
                return 2

        else:
            assert False, "verify token_ids was list, not int"

    def expect_remain_len_idx(self, threshold_list: list[float]) -> int:
        # remain length is given in idx, not real length eg. if 4 -> idx 0, 8 -> idx 1 ...
        remain_length = -1

        for idx in range(len(self.predicted_len_prob)):
            if self.predicted_len_prob[idx] >= threshold_list[idx]:
                remain_length = idx
                return remain_length
            
        return remain_length

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id].length
        return num_tokens

    @property
    def use_structured_output(self) -> bool:
        return self.sampling_params.guided_decoding is not None

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: Optional[float] = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
