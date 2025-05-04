from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker


tracker = EvaluationTracker(
    output_dir="results/",    # where to write JSON/parquet
    save_details=False,        # include per-sample logs
    push_to_hub=False,        # toggle uploading to HF Hub
)

params = PipelineParameters(
    launcher_type=ParallelismManager.ACCELERATE,  # use Accelerate
    max_samples=100,        # limit total examples
)

tasks = "leaderboard|truthfulqa:mc|0|0"

model_cfg = TransformersModelConfig(
    model_name="gpt2",       # HF model ID
    revision="main",          # branch or tag
    add_special_tokens=True,  # if your prompts need extra tokens
    dtype="float32",          # torch dtype
    use_chat_template=False,  # zero-shot style
)

# 5. Build and run the pipeline
pipeline = Pipeline(
    tasks=tasks,
    pipeline_parameters=params,
    evaluation_tracker=tracker,
    model_config=model_cfg,
)

pipeline.evaluate()            # run the eval
pipeline.save_and_push_results()  # write files (and push if enabled)
pipeline.show_results() 