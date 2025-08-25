import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./alpha-448101-282bc1b884cd.json"

from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jaxpt.models.smol_lm import SmolLM, SmolLM_Config, convert_to_hf
from jaxpt.checkpointers import load_checkpoint_from_gcloud

from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import (
    TransformersModelConfig,
)
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics



def main():
    output_dir = "results/"
    save_details = True
    push_to_hub = False
    max_samples = None
    launcher_type = ParallelismManager.ACCELERATE


    tasks = [
        "lighteval|triviaqa|0|0",
        "lighteval|arc:easy|0|0",
        "leaderboard|arc:challenge|0|0",
        "helm|piqa|0|0",
        "leaderboard|hellaswag|0|0",
        "helm|openbookqa|0|0",
        "leaderboard|winogrande|0|0",
    ]

    tasks = [
        "custom|hellaswag|0|1",
    ]

    model_cfg = TransformersModelConfig(
        model_name="HuggingFaceTB/SmolLM-135M",
        dtype="bfloat16",
        add_special_tokens=True,
        max_length=2048,
        batch_size=16,
    )

    tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
    )

    params = PipelineParameters(
        launcher_type=launcher_type,
        max_samples=max_samples,
        custom_tasks_directory="/Users/vikram/dev/jaxpt/src/jaxpt/evals/lighteval/tasks/"
    )

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
