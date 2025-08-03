from jaxpt.models.tiny_moe import Tiny_MoE, Config_Tiny_MoE

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import (
    Batch,
    ModelResponse,
)

class Lighteval_Tiny_MoE(LightevalModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def greedy_until(self, requests, max_tokens=None, stop_sequences=None) -> list[ModelResponse]:
        # Implement generation logic
        pass

    def loglikelihood(self, requests, log=True) -> list[ModelResponse]:
        # Implement loglikelihood computation
        pass

    def loglikelihood_rolling(self, requests) -> list[ModelResponse]:
        # Implement rolling loglikelihood computation
        pass

    def loglikelihood_single_token(self, requests) -> list[ModelResponse]:
        # Implement single token loglikelihood computation
        pass