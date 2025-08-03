import os 

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import argparse
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.models.tiny_moe import Tiny_MoE, Tiny_MoE_Config
from jaxpt.checkpointers import load_checkpoint_from_gcloud
from jaxpt.utils import count_params

from transformers import AutoTokenizer
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import (
    Batch,
    ModelResponse,
)
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker


devices = jax.devices()
num_devices = len(devices)
print("Available devices:", num_devices)

requested_device = "gpu"

jax.config.update("jax_platform_name", requested_device) # Make sure we're using the GPU

device = jax.default_backend()
if device != requested_device:
    warnings.warn(f"not using {requested_device}. Using {device}")
else:
    print(f"using {device}")

jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32") # Set the default precision for matrix multiplication

mesh = jax.sharding.Mesh(jax.devices(), ["devices"])


class Lighteval_Tiny_MoE(LightevalModel):
    def __init__(self, config):
        self.config = config
        config = Tiny_MoE_Config(
                        name="Tiny_MoE",
                        dtype=jnp.bfloat16, \
                        vocab_size=49152,
                        n_layer=30,
                        block_size=2048,
                        n_head=9,
                        n_kv_head=3,
                        n_mlp_hidden=1536,
                        expert_weight_priority=False,
                        load_factor=2.0,
                        sdpa_implementation="cudnn" if device=="gpu" else "xla")
        with mesh:
            self.model = self._create_model(config) 
        self._tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        self._add_special_tokens = False
        self._max_length = config.block_size

        self.model_info = ModelInfo(
            model_name=config.name,
            model_sha="",
            model_dtype=config.dtype,
            model_size=count_params(self.model) * 2,
        )


    def _create_model(self, config):
        key = jax.random.PRNGKey(1337)
        rngs = nnx.Rngs(key)
        return load_checkpoint_from_gcloud(
            Tiny_MoE, config, Path().absolute(), 
            "alpha_training_runs", 
            "run_20250729_berne_abraham", 
            329971, 
            rngs
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def greedy_until(self, requests, max_tokens=None, stop_sequences=None) -> list[ModelResponse]:
        # Implement generation logic
        print(requests)
        return list() 

    def loglikelihood(self, requests, log=True) -> list[ModelResponse]:
        # Implement loglikelihood computation
        print(requests)
        return list()

    def loglikelihood_rolling(self, requests) -> list[ModelResponse]:
        # Implement rolling loglikelihood computation
        pass

    def loglikelihood_single_token(self, requests) -> list[ModelResponse]:
        # Implement single token loglikelihood computation
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LightEval evaluation pipeline"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to write JSON/parquet results",
    )
    parser.add_argument(
        "--save_details",
        action="store_true",
        help="Include per-sample logs in output",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload results to the Hugging Face Hub",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the total number of examples to evaluate",
    )
    parser.add_argument(
        "--launcher_type",
        type=str,
        default=ParallelismManager.ACCELERATE,
        help="Parallelism launcher type (e.g., ACCELERATE)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        #default="lighteval|arc:easy|0|0,leaderboard|arc:challenge|0|0,helm|piqa|0|0,helm|siqa|0|0,leaderboard|hellaswag|0|0,helm|openbookqa|0|0,leaderboard|winogrande|0|0,lighteval|triviaqa|0|0,lighteval|race:high|0|0",
        default="leaderboard|hellaswag|0|0",
        help="Comma-separated list of tasks to run",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Pretrained model identifier or path for TransformersModelConfig",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    key = jax.random.PRNGKey(1337)
    rngs = nnx.Rngs(key)
    config = Tiny_MoE_Config(
                     name="Tiny_MoE",
                     dtype=jnp.bfloat16, \
                     vocab_size=49152,
                     n_layer=30,
                     block_size=2048,
                     n_head=9,
                     n_kv_head=3,
                     n_mlp_hidden=1536,
                     expert_weight_priority=False,
                     load_factor=2.0,
                     sdpa_implementation="cudnn" if device=="gpu" else "xla")

    output_dir = Path("/workspace/").absolute()

    model_cfg = CustomModelConfig(
        model_name="Tiny_MoE",
        model_definition_file_path=str(Path().absolute() / "src" / "jaxpt"/ "evals" / "lighteval_tiny_moe.py")
    )

    tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=args.save_details,
        push_to_hub=args.push_to_hub,
    )

    params = PipelineParameters(
        launcher_type=args.launcher_type,
        max_samples=args.max_samples,
    )

    tasks = args.tasks.split(",")

    pipeline = Pipeline(
        tasks=",".join(tasks),
        pipeline_parameters=params,
        evaluation_tracker=tracker,
        model_config=model_cfg,
    )

    pipeline.evaluate()  # run the eval
    # pipeline.save_and_push_results()  # write files (and push if enabled)
    pipeline.show_results()


if __name__ == "__main__":
    main()
