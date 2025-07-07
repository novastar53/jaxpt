import os

from google.cloud import storage


def save_checkpoint(m, output_dir, run_dirname, step):
    checkpoint_dirpath = (
        output_dir / m.config.name / "checkpoints" / run_dirname
    )
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    m.save_checkpoint(checkpoint_dirpath / f"checkpoint-{step}.pt")


def load_checkpoint(model, output_dir, config, run_dirname, step, rngs):
    checkpoint_path = (
        output_dir
        / config.name
        / "checkpoints"
        / run_dirname
        / f"checkpoint-{step}.pt"
    )
    m = model.from_checkpoint(checkpoint_path, rngs, config)
    return m


def load_checkpoint_from_gcloud(
    model, config, output_dir, bucket_name, run_dirname, step, rngs
):
    client = storage.Client()
    prefix = f"{config.name}/checkpoints/{run_dirname}/checkpoint-{step}.pt/"
    checkpoint_path = (
        output_dir
        / config.name
        / "checkpoints"
        / run_dirname
        / f"checkpoint-{step}.pt/"
    )
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel_path = blob.name[len(prefix) :]
        dst_path = os.path.join(checkpoint_path, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        blob.download_to_filename(dst_path)
    m = model.from_checkpoint(checkpoint_path, rngs, config)
    return m
