import os

import flax.nnx as nnx
import orbax.checkpoint as ocp
from google.cloud import storage


def save_checkpoint(m, output_dir, run_dirname, step):
    checkpoint_dirpath = (
        output_dir / m.config.name / "checkpoints" / run_dirname
    )
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dirpath / f"checkpoint-{step}.pt"
    print(f"Saving model checkpoint to {checkpoint_path}")
    m.save_checkpoint(checkpoint_path)


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
    try:
        return load_checkpoint(model, output_dir, config, run_dirname, step, rngs)
    except:
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


def save_optimizer_state(output_dir, run_dirname, config, optimizer):
  state_dirpath = (
    output_dir / config.name / "optimizer_checkpoints" / run_dirname
  )
  state_dirpath.mkdir(parents=True, exist_ok=True)
  _, state = nnx.split(optimizer)
  cp = ocp.StandardCheckpointer()
  print(f"Saving optimizer state to {state_dirpath}/step-{optimizer.step.value.item()}")
  cp.save(state_dirpath / f"step-{optimizer.step.value.item()}", state)
  cp.wait_until_finished()


def load_optimizer_state(config, optimizer, output_dir, run_dirname, step):
  cp = ocp.StandardCheckpointer()
  graphdef, state = nnx.split(optimizer)
  path = output_dir / config.name / "optimizer_checkpoints" / run_dirname / f"step-{step}"
  state = cp.restore(path, target=state)
  print(f"loading_optimizer_state from {path}")
  optimizer = nnx.merge(graphdef, state)
  return optimizer

