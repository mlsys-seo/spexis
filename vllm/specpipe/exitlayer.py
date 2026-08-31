# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.loader import (DefaultModelLoader,
                                                     get_model_loader)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .exit_models.llama_adapter import AdapterModel as LlamaAdapterModel
from .exit_models.qwen_adapter import AdapterModel as QwenAdapterModel


class ExitLayerModel(nn.Module):
    """Per-stage wrapper around the early-exit (draft) adapter model.

    Selects the adapter class by ``exit_model_type``, loads its weights
    lazily via ``DefaultModelLoader.load_exitmodel``, and exposes the
    forward / compute_logits / sample interface the model runner expects.

    Note: the exit model shares the process-global torch.compile machinery
    (cudagraph memory pool, dynamo cache) with the target model. Keep
    ``@support_torch_compile`` off the adapters unless capturing two models
    in one process has been validated.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        pp_rank: int,
        prefix: str = "",
        dtype: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config

        self.load_config = vllm_config.load_config
        self.specpipe_config = vllm_config.specpipe_config

        # lazy initializing AdapterModel
        self.hf_config = vllm_config.specpipe_config.exit_hf_config[pp_rank]
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config

        self.pp_rank = pp_rank

        self.model_class = None
        if self.specpipe_config.exit_model_type == 'qwen':
            self.model_class=QwenAdapterModel
        elif self.specpipe_config.exit_model_type == 'llama':
            self.model_class=LlamaAdapterModel
        else:
            assert False, f"Didn't implemented {self.specpipe_config.exit_model_type} adapter model"

        self.unpadded_vocab_size = self.hf_config.vocab_size

        logit_scale = getattr(self.hf_config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.hf_config.vocab_size,
                                                logit_scale)

        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], self.hf_config.hidden_size))

    def load_model(self):
        loader = get_model_loader(self.load_config)

        assert type(loader) == DefaultModelLoader, (
            f"loader was not DefaultModelLoader: {type(loader)}")

        self.model = loader.load_exitmodel(
            vllm_config=self.vllm_config,
            model_config=self.specpipe_config.exit_model_config[self.pp_rank],
            device_config=self.device_config,
            pp_rank=self.pp_rank,
            model_class=self.model_class,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)

        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
