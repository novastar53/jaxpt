import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./alpha-448101-282bc1b884cd.json"

from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jaxpt.models.smol_lm import SmolLM, SmolLM_Config, convert_to_hf
from jaxpt.checkpointers import load_checkpoint_from_gcloud


from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import (
    TransformersModelConfig,
)
from lighteval.logging.evaluation_tracker import EvaluationTracker
import argparse


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
    config = SmolLM_Config(dtype=jnp.float16, \
                        vocab_size=49152,
                        n_embed=576,
                        n_head=9,
                        n_kv_head=3,
                        n_mlp_hidden=1536,
                        sdpa_implementation="xla")

    output_dir = Path("/workspace/").absolute()
    #m = load_checkpoint_from_gcloud(SmolLM, config, output_dir, "alpha_training_runs", "run_20250721_wrbyxb", "10000", rngs)
    #m = convert_to_hf(m)
    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")

    model_cfg = TransformersModelConfig(
        model_name="HuggingFaceTB/SmolLM-135M",
        dtype="bfloat16",
        add_special_tokens=True,
        max_length=2048,
        batch_size=16,
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
        model=m,
        model_config=model_cfg,
    )

    pipeline.evaluate()  # run the eval
    # pipeline.save_and_push_results()  # write files (and push if enabled)
    pipeline.show_results()


if __name__ == "__main__":
    main()
