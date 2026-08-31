from collections.abc import Iterable
from typing import List, Tuple, Set
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.model_loader.loader import (DefaultModelLoader,
                                                     get_model_loader)
from vllm.sequence import IntermediateTensors

class LengthPredictor(nn.Module):
    """
    Predictor model that has independent prediction nodes for each tau (threshold).
    Instead of a single score + fixed threshold approach,
    it has K independent heads that each determine 'is the remaining length <= tau?'.
    """
    def __init__(
        self, 
        in_dim: int, 
        thresholds_list: List[int], 
        hidden: int = 512, 
        dropout_p: float = 0.1, 
        prefix: str = "",
        ):
        
        super().__init__()

        # Metadata
        self.prefix = prefix
        self.num_classes = len(thresholds_list)
        self.thresholds_list = thresholds_list

        # 1. Feature Extractor (Backbone)
        # Input: LLM hidden states (in_dim)
        # Output: Feature vector (hidden)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.LayerNorm(hidden),

            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.LayerNorm(hidden),
        )
        
        # 2. Independent Binary Heads
        # Each head outputs a logit for P(rem <= tau_k)
        self.head = nn.Linear(hidden, self.num_classes)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None
        )

        return loader.load_weights(
            (name, loaded_weight) for name, loaded_weight in weights)
    
    def forward(self, num_input_tokens: int,
                intermediate_tensors: IntermediateTensors) -> torch.Tensor:
        """
        Args:
            num_input_tokens: number of valid rows in the stage buffers.
            intermediate_tensors: stage output holding "hidden_states" and
                "residual", each (num_tokens, in_dim).
        Returns:
            probs: (num_input_tokens, K) sigmoid probability per head,
                P(remaining length <= tau_k).
        """
        x = (intermediate_tensors["hidden_states"][:num_input_tokens] +
             intermediate_tensors["residual"][:num_input_tokens])

        features = self.backbone(x)
        logits = self.head(features)  # (B, K), one logit per threshold head
        probs = torch.sigmoid(logits)

        return probs

class LengthPredictorWrapper(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        pp_rank: int,
        prefix: str = "",
        dtype: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config

        # lazy initializing LengthPredictor
        self.model_config_dict = vllm_config.specpipe_config.length_predictor_config
        self.model_path = vllm_config.specpipe_config.length_predictor_path

        self.load_config = vllm_config.load_config
        self.device_config = vllm_config.device_config

        self.pp_rank = pp_rank

        self.model_class = LengthPredictor

    def load_model(self):
        # self.model is created lazily here (load_length_predictor), not
        # in __init__, mirroring ExitLayerModel.
        loader = get_model_loader(self.load_config)

        assert type(loader) == DefaultModelLoader, (
            f"loader was not DefaultModelLoader: {type(loader)}")

        self.model = loader.load_length_predictor(
            vllm_config=self.vllm_config,
            model_config_dict=self.model_config_dict,
            device_config=self.device_config,
            pp_rank=self.pp_rank,
            model_class=self.model_class,
        )
