import os

from pathlib import Path
import argparse
import warnings

import jax
import flax
import jax.numpy as jnp
import flax.nnx as nnx
from jaxpt.models.gpt import GPT2, GPT2_Config 
from jaxpt.checkpointers import save_checkpoint, load_checkpoint, load_checkpoint_from_gcloud
from jaxpt.utils import create_sharded_model


from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import (
    TransformersModelConfig,
)
from lighteval.logging.evaluation_tracker import EvaluationTracker

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./alpha-448101-282bc1b884cd.json"

# Hardware setup
print("JAX version:", jax.__version__)
print("Flax version", flax.__version__)

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

output_dir = Path("/workspace/alpha_training_runs") 

mesh = jax.sharding.Mesh(jax.devices(), ["devices"])

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
        default="lighteval|arc:easy|0|0," #leaderboard|arc:challenge|0|0,helm|piqa|0|0,helm|siqa|0|0,leaderboard|hellaswag|0|0,helm|openbookqa|0|0,leaderboard|winogrande|0|0,lighteval|triviaqa|0|0,lighteval|race:high|0|0",
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

    default = jax.random.key(1337)
    gate_noise = jax.random.key(42)
    rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
    config = Tiny_MoE_Config(dtype=jnp.bfloat16, \
                        vocab_size=49152,
                        n_layer=4,
                        block_size=128,
                        n_head=9,
                        n_kv_head=3,
                        n_mlp_hidden=1536,
                        sdpa_implementation="cudnn" if device=="gpu" else "xla")
    nnx.display(config)
    with mesh:
        #flax_model = load_checkpoint_from_gcloud(Tiny_MoE, config, output_dir, "alpha_training_runs", "run_20250722_uhixrg", 20000, rngs)
        flax_model = create_sharded_model(Tiny_MoE, config, rngs)

    model_cfg = TransformersModelConfig(
        model_name="HuggingFaceTB/GPT2-",
        dtype="float32",
        add_special_tokens=True,
        max_length=2048,
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
        model=flax_model,
        model_config=model_cfg,
    )

    pipeline.evaluate()  # run the eval
    # pipeline.save_and_push_results()  # write files (and push if enabled)
    pipeline.show_results()


if __name__ == "__main__":
    main()
