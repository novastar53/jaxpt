from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run LightEval evaluation pipeline")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to write JSON/parquet results")
    parser.add_argument("--save_details", action="store_true",
                        help="Include per-sample logs in output")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Upload results to the Hugging Face Hub")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit the total number of examples to evaluate")
    parser.add_argument("--launcher_type", type=str,
                        default=ParallelismManager.ACCELERATE,
                        help="Parallelism launcher type (e.g., ACCELERATE)")
    parser.add_argument("--tasks", type=str,
                        default="lighteval|arc:easy|0|0,leaderboard|arc:challenge|0|0,helm|piqa|0|0,helm|siqa|0|0,leaderboard|hellaswag|0|0,helm|openbookqa|0|0,leaderboard|winogrande|0|0,lighteval|triviaqa|0|0,lighteval|race:high|0|0",
                        help="Comma-separated list of tasks to run")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Pretrained model identifier or path for TransformersModelConfig")
    return parser.parse_args()

def main():

    args = parse_args()

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

    model_cfg = TransformersModelConfig(model_name_or_path=args.model_name_or_path)

    # 5. Build and run the pipeline
    pipeline = Pipeline(
        tasks=",".join(tasks),
        pipeline_parameters=params,
        evaluation_tracker=tracker,
        model_config=model_cfg,
    )

    pipeline.evaluate()            # run the eval
    pipeline.save_and_push_results()  # write files (and push if enabled)
    pipeline.show_results() 

if __name__ == "__main__":
    main()