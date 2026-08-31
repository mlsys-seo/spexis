# SPDX-License-Identifier: Apache-2.0
"""GPU model runner for speculative pipeline parallelism (Spexis).

Subclasses the upstream :class:`GPUModelRunner`, which stays responsible
for everything a normal run does. This class adds the pieces the Spexis
pipeline needs:

* the early-exit (draft) model, and optionally the length predictor, for
  the pipeline stage this rank owns;
* ``_update_states`` / ``_prepare_inputs`` overrides that roll back the
  tokens of a rejected speculation before building the batch;
* ``execute_target_model`` / ``return_hidden_state`` /
  ``execute_exitlayer_model``, the entry points the Ray compiled DAG
  calls to run the target model and the exit layer concurrently
  (see vllm/executor/ray_utils.py);
* dummy-run variants so memory profiling covers the exit layer.
"""

from typing import Optional, Union

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.specpipe.bool_token import BoolToken
from vllm.specpipe.exitlayer import ExitLayerModel
from vllm.specpipe.length_predictor import LengthPredictorWrapper
from vllm.specpipe.sched_output import SpecpipeSchedulerOutput
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class SpecpipeGPUModelRunner(GPUModelRunner):
    """Model runner that runs the target model and an exit layer together."""

    enable_specpipe = True

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self.sppp_len_pred = self.specpipe_config.sppp_len_pred

        # Build the exit layer only on the ranks configured to speculate.
        # NOTE: rank_in_group is the pipeline stage index; combining this
        # with tensor parallelism has not been validated.
        pp_rank = get_pp_group().rank_in_group
        if self.specpipe_config.exit_layer[pp_rank]:
            self.exit_model = ExitLayerModel(self.vllm_config, pp_rank)
            if self.sppp_len_pred:
                self.length_predictor = LengthPredictorWrapper(
                    self.vllm_config, pp_rank)

    def _load_specpipe_models(self) -> None:
        if hasattr(self, "exit_model"):
            logger.info("Loading exit layer model...")
            self.exit_model.load_model()
        if hasattr(self, "length_predictor"):
            logger.info("Loading length predictor model...")
            self.length_predictor.load_model()

    def _profile_specpipe_run(self):
        if not hasattr(self, "exit_model"):
            return None, None
        # NOTE: the sampler dummy run is approximate; it does not model
        # speculative decoding combined with an exit layer.
        sp_hidden_states = self._dummy_specpipe_run(self.max_num_tokens)
        return sp_hidden_states, self._dummy_sp_sampler_run(sp_hidden_states)

    def _update_states(self,
                                scheduler_output: "SpecpipeSchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.

        # NOTE: with pipeline micro-batching the two in-flight batches
        # alternate, so requests churn in and out of the batch every
        # step; the state itself is preserved across the removal.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]  # type: CachedRequestState

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens

            # Modify Spec failed requests
            if req_data.spec_result is False:
                # Modify gpu token states
                req_state.output_token_ids = req_state.output_token_ids[:req_data
                                                                        .
                                                                        fixed_idx] + req_data.new_token_ids
                # Modify the block IDs.
                if not req_data.resumed_from_preemption:
                    # Append the new blocks to the existing block IDs.
                    req_state.block_ids.extend(req_data.new_block_ids)
                else:
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    req_state.block_ids = req_data.new_block_ids

                req_index = self.input_batch.req_id_to_index.get(req_id)
                if req_index is None:
                    # The request is not in the persistent batch.
                    # The request was either preempted and resumed later, or was not
                    # scheduled in the previous step and needs to be added again.
                    req_ids_to_add.append(req_id)
                    continue
            # Add new tokens for spec successed requests
            else:
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec decode tokens.
                num_new_tokens = (num_computed_tokens +
                                  len(req_data.new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(
                        req_data.new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        req_data.new_token_ids[-num_new_tokens:])
                # Update the block IDs.
                if not req_data.resumed_from_preemption:
                    # Append the new blocks to the existing block IDs.
                    req_state.block_ids.extend(req_data.new_block_ids)
                else:
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    req_state.block_ids = req_data.new_block_ids

                req_index = self.input_batch.req_id_to_index.get(req_id)
                if req_index is None:
                    # The request is not in the persistent batch.
                    # The request was either preempted and resumed later, or was not
                    # scheduled in the previous step and needs to be added again.
                    req_ids_to_add.append(req_id)
                    continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)

            # Add/Modify new_token_ids to token_ids_cpu.
            if req_data.spec_result is False:
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(
                    req_data.new_token_ids)
                # correct token from target output
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:start_token_index +
                    1] = req_data.new_token_ids
                # erase flushed tokens after wrong token
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index + 1:start_token_index +
                    self.parallel_config.pipeline_parallel_size + 1] = [0]
                self.input_batch.num_tokens_no_spec[
                    req_index] = start_token_index + 1
            else:
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(
                    req_data.new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = req_data.new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[FlashAttentionMetadata, torch.Tensor,
               Optional[SpecDecodeMetadata]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Some attention backends (namely MLA) may want to separate requests
        # based on if the attention computation will be compute-bound or
        # memory-bound. This gives them a hook to do that.
        modified_batch = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)
        if modified_batch:
            self.input_batch.refresh_sampling_metadata()

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        # Prepare for cascade attention if enabled & beneficial.
        common_prefix_len = 0
        if self.cascade_attn_enabled:
            common_prefix_len = self._compute_cascade_attn_prefix_len(
                num_scheduled_tokens,
                scheduler_output.num_common_prefix_blocks,
            )

        attn_metadata = self.attn_metadata_builder.build(
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            common_prefix_len=common_prefix_len,
        )

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = attn_metadata.query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return attn_metadata, logits_indices, spec_decode_metadata

    @torch.inference_mode()
    def execute_target_model(
        self,
        scheduler_output: "SpecpipeSchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        torch.cuda.nvtx.range_push(msg="execute_target_model")

        
        self._update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # Prepare the decoder inputs.
        attn_metadata, logits_indices, spec_decode_metadata = (
            self._prepare_inputs(scheduler_output))

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = num_scheduled_tokens
        attn_metadata.num_input_tokens = num_input_tokens

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]

            # Copy in the CPU-computed embeddings, then hand the model
            # the padded view so cudagraph batch sizes still line up.
            # TODO: avoid this copy.
            self.inputs_embeds[:num_scheduled_tokens].copy_(
                scheduler_output.embeddings)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if self.specpipe_config.enable_specpipe and get_pp_group().is_first_rank:
                self.cached_tensors = hidden_states
            torch.cuda.nvtx.range_pop()
            return BoolToken({"tok":True})

        torch.cuda.nvtx.range_pop()
        return BoolToken({"tok":False})

    def return_hidden_state(
            self,
            token: BoolToken
    ):
        # for k, v in token.items():
        #     token_out = v
        # assert token_out, "token was False"
        torch.cuda.nvtx.mark(msg="return_hidden_state")
        return self.cached_tensors

    @torch.inference_mode()
    def execute_exitlayer_model(
        self,
        scheduler_output: "SpecpipeSchedulerOutput",
        token: BoolToken
        # intermediate_tensors: "IntermediateTensors",
    ) -> Union[ModelRunnerOutput, torch.Tensor]:

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        intermediate_tensors = self.cached_tensors

        torch.cuda.nvtx.range_push(msg="execute_exitlayer_model")
        # Prepare the decoder inputs.
        attn_metadata, logits_indices, spec_decode_metadata = (
            self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = num_scheduled_tokens
        attn_metadata.num_input_tokens = num_input_tokens

        # For text-only models, we use token ids as input.
        # While it is possible to use embeddings as input just like the
        # multimodal models, it is not desirable for performance since
        # then the embedding layer is not included in the CUDA graph.
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        assert intermediate_tensors is not None
        assert self.intermediate_tensors is not None

        for k, v in intermediate_tensors.items():
            self.intermediate_tensors[k][:num_input_tokens].copy_(
                v[:num_input_tokens], non_blocking=True)

        intermediate_tensors = IntermediateTensors({
            k: v[:num_input_tokens]
            for k, v in self.intermediate_tensors.items()
        })

        if self.sppp_len_pred:
            torch.cuda.nvtx.range_push(msg="execute_predl_model")
            length_probs = self.length_predictor.model(num_input_tokens, intermediate_tensors)
            length_probs = length_probs[:num_scheduled_tokens]
            sample_length_probs = length_probs[logits_indices]
            torch.cuda.nvtx.range_pop()


        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.exit_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        hidden_states = hidden_states[:num_scheduled_tokens]
        sample_hidden_states = hidden_states[logits_indices]
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(msg='lm_head')
        logits = self.exit_model.compute_logits(sample_hidden_states, None)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(msg='remaining')
        # Confidence of the drafted token = its softmax probability;
        # the scheduler multiplies these along a speculation run.
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        top1s = torch.max(probs, dim=-1, keepdim=True)
        top1_probs = top1s.values
        top1_token_ids = top1s.indices


        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata


        

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        # logprobs_tensors = sampler_output.logprobs_tensors
        # logprobs_lists = logprobs_tensors.tolists() \
        #     if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states,
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = top1_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
            valid_top1_probs = top1_probs.tolist()
            if self.sppp_len_pred:
                valid_length_probs = sample_length_probs.tolist()
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
            valid_top1_probs[i].clear()
            if self.sppp_len_pred:
                valid_length_probs[i].clear()


        if not self.use_spec_decode:
            # Speculative decoding is not enabled.
            spec_token_ids = None

        torch.cuda.nvtx.range_pop()

        if not self.sppp_len_pred:
            valid_length_probs = None

        
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            spec_confidence=valid_top1_probs,
            length_probs=valid_length_probs,
        )

    @torch.inference_mode()
    def _dummy_specpipe_run(
        self,
        num_tokens: int,
    ) -> torch.Tensor:

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = max_num_reqs if num_tokens >= max_num_reqs else num_tokens
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.exit_model
            if self.is_multimodal_model:
                assert False, "Multi-Modal model is not supported for specpipe"
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.exit_model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device))
            intermediate_tensors = IntermediateTensors({
                k: v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            })

            with set_forward_context(None,
                                     self.vllm_config,
                                     num_tokens=num_tokens):
                hidden_states = model(  # [total_tokens_in_batch, hidden]
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states[logit_indices]  # [batch_size, hidden]

    @torch.inference_mode()
    def _dummy_sp_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        logits = self.exit_model.compute_logits(hidden_states, None)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full(
            (num_reqs, ), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            min_p=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            min_tokens={},
            logit_bias=[None for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
        )
        try:
            sampler_output = self.exit_model.sample(
                logits=logits, sampling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        if self.use_spec_decode:
            raise NotImplementedError(
                "speculative decoding is not supported together "
                "with speculative pipelining")
        return sampler_output
