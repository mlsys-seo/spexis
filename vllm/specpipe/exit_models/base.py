# SPDX-License-Identifier: Apache-2.0
"""Architecture-agnostic body of the Spexis early-exit adapter.

An exit adapter turns the intermediate ``(hidden_states, residual)`` that a
pipeline stage ships to the next stage into draft-token logits:

    initial RMSNorm over (hidden + residual)
      -> Linear bridge (trained, maps stage output into adapter space)
      -> N attention-only decoder layers (no FFN; they continue the target
         model's layer numbering so checkpoint weight names line up)
      -> ParallelLMHead

Architecture-specific files (llama_adapter, qwen_adapter) provide the
attention/decoder-layer classes and subclass :class:`ExitAdapterModel`
by setting ``decoder_layer_cls``.
"""
from typing import Iterable, Optional, Set, Tuple, Type

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors


class ExitAdapterModel(nn.Module):
    """Common exit-adapter body; subclasses set ``decoder_layer_cls``."""

    decoder_layer_cls: Type[nn.Module]

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "adaptermodel",
        pp_rank: int,
    ):
        super().__init__()

        self.config = vllm_config.specpipe_config.exit_hf_config[pp_rank]
        self.cache_config = vllm_config.cache_config

        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        # Index of the target-model layer this stage exits after. Adapter
        # layers continue that numbering (see prefix below) so that the
        # checkpoint's weight names map onto the right modules.
        self.exit_layer = (pp_rank + 1) * (
            vllm_config.model_config.hf_config.num_hidden_layers //
            vllm_config.parallel_config.pipeline_parallel_size)

        self.layers = nn.ModuleList([
            self.decoder_layer_cls(
                self.config,
                self.cache_config,
                prefix=f"{prefix}.layers.{i + self.exit_layer}")
            for i in range(self.config.num_hidden_layers)
        ])

        self.mlp = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.initial_layernorm = RMSNorm(self.config.hidden_size,
                                         eps=self.config.rms_norm_eps)

        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if self.config.tie_word_embeddings:
            raise NotImplementedError(
                "tied word embeddings are not supported for exit adapters")

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )

        return loader.load_weights(
            (name, loaded_weight) for name, loaded_weight in weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

        hidden_states, _ = self.initial_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(positions, hidden_states)

        return hidden_states
