import os

import tqdm 

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

from typing import Union, Optional
import argparse
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import numpy as np

from jaxpt.models.tiny_moe import Tiny_MoE, Tiny_MoE_Config
from jaxpt.checkpointers import load_checkpoint_from_gcloud
from jaxpt.utils import count_params

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset, LoglikelihoodSingleTokenDataset
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)

TokenSequence = Union[list[int], jax.Array, BatchEncoding]

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

STARTING_BATCH_SIZE = 512

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


        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM-135M"
        )
        self._tokenizer = tokenizer
        self.pairwise_tokenization = False
        self._add_special_tokens = False
        self._max_length = config.block_size

        self.model_info = ModelInfo(
            model_name=config.name,
            model_sha="",
            model_dtype=str(config.dtype),
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

    def greedy_until(self, requests, max_tokens=None, stop_sequences=None) -> list[GenerativeResponse]:
        # Implement generation logic
        print(requests)
        return list() 

    def loglikelihood(self, requests, log=True) -> list[LoglikelihoodResponse]:
        # Implement loglikelihood computation
        print(requests)
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode_jax(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice, pairwise=self.pairwise_tokenization
                )

        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE
        res = []


        for split in tqdm(dataset.splits_iterator(), disable=self.disable_tqdm):
            context_enc = split[0].tokenized_context
            continuation_enc = split[0].tokenized_continuation
            if rolling:  # we take all the sequence in rolling mode
                max_input_length = len(context_enc + continuation_enc)
            else:  # in normal mode, we left cut the context if needed
                max_input_length = max(min(self.max_length, len(context_enc + continuation_enc) - 1), 1)

            batch_size = self._get_batch_size(
                override_bs=self.config.batch_size,
                max_input_length=max_input_length,
                starting_batch_size=starting_batch_size,
            )
            starting_batch_size = batch_size * 2

            dataloader = DataLoader(split, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm):
                prepared_batch = self.prepare_batch_logprob(
                    batch,
                    padding_length=max_input_length,
                    max_context=max_input_length,
                )

                model_output = self._model_call(prepared_batch.input_ids)
                logits = F.log_softmax(model_output, dim=-1)  # [batch, padding_length, vocab]

                logits_sum = []
                max_equals = []
                batch_cont_tokens = []
                for cur_request, cur_logits, inplen in zip(batch, logits, prepared_batch.input_lengths):
                    cont_toks = torch.tensor(cur_request.tokenized_continuation, dtype=torch.long, device=self.device)
                    contlen = cont_toks.shape[0]
                    # We only look at the continuation tokens
                    if contlen > inplen:
                        # Continuation is longer than the input size, we are in rolling mode (only continuation)
                        cur_logits = cur_logits.unsqueeze(0).to(self.device)  # [1, seq, vocab]
                        cont_toks = cont_toks[:inplen].unsqueeze(0).to(self.device)  # [1, seq]
                    else:
                        cur_logits = (
                            cur_logits[inplen - contlen : inplen].unsqueeze(0).to(self.device)
                        )  # [1, seq, voc]
                        cont_toks = cont_toks.unsqueeze(0).to(self.device)  # [1, seq]

                    # Check if per-token argmax is exactly equal to continuation
                    greedy_tokens = cur_logits.argmax(dim=-1).to(self.device)
                    # Sometimes the continuation is longer than allowed by the model, we only look at the first tokens
                    max_equal = (greedy_tokens == cont_toks).all().squeeze(0).to(self.device)

                    # Obtain log-probs at the corresponding continuation token indices
                    cur_logits = torch.gather(cur_logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    logits_sum.append(cur_logits.sum())
                    max_equals.append(max_equal)
                    batch_cont_tokens.append(cont_toks)

                # Sync all
                # Need reshaping before gather
                batched_inputs, len_inputs = self.pad_and_gather(prepared_batch.input_ids)
                max_cont_tokens_length = max(len(c[0]) for c in batch_cont_tokens)
                # These are the true lengths of the continuation tokens, we have to save them to be able to removed padding tokens from the generated tokens.
                batch_cont_token_lengths = torch.tensor([c.shape[1] for c in batch_cont_tokens], device=self.device)
                batch_cont_tokens = torch.cat(
                    [
                        F.pad(c, (0, max_cont_tokens_length - c.shape[1], 0, 0), value=self.tokenizer.pad_token_id)
                        for c in batch_cont_tokens
                    ],
                    dim=0,
                )
                batch_cont_tokens, _ = self.pad_and_gather(batch_cont_tokens)
                # Can be gathered as such
                logits = torch.tensor(logits_sum, device=self.device)
                max_equal = torch.tensor(max_equals, device=self.device)
                batch_truncated = torch.tensor(prepared_batch.truncated, device=self.device)
                batch_padded = torch.tensor(prepared_batch.padded, device=self.device)
                if self.accelerator:
                    logits = self.accelerator.gather_for_metrics(logits)
                    max_equal = self.accelerator.gather_for_metrics(max_equal)
                    batch_truncated = self.accelerator.gather_for_metrics(batch_truncated)
                    batch_padded = self.accelerator.gather_for_metrics(batch_padded)
                    batch_cont_token_lengths = self.accelerator.gather_for_metrics(batch_cont_token_lengths)

                for logit, cont_tokens, maxe, batched_input, trunc, padded, len_input, len_token in zip(
                    logits,
                    batch_cont_tokens,
                    max_equal,
                    batched_inputs,
                    batch_truncated,
                    batch_padded,
                    len_inputs,
                    batch_cont_token_lengths,
                ):
                    # Filter out padding tokens from input_tokens and generated_tokens
                    input_tokens = batched_input[:len_input].cpu().tolist()
                    generated_tokens = cont_tokens[:len_token].cpu().tolist()

                    answer = LoglikelihoodResponse(
                        # todo: we might want to store the logits unsummed
                        result=(float(logit.sum()), bool(maxe)) if return_bool_score else float(logit.sum()),
                        input_tokens=input_tokens,
                        generated_tokens=generated_tokens,
                        truncated_tokens_count=trunc.cpu().item(),
                        padded_tokens_count=padded.cpu().item(),
                    )
                    res.append(answer)

                # Clean up GPUs
                del model_output
                del logits
                del batched_inputs
                del batch_truncated
                del batch_padded

        return dataset.get_original_order(res)


    def loglikelihood_rolling(self, requests) -> list[LoglikelihoodResponse]:
        raise NotImplementedError

    def loglikelihood_single_token(self, requests) -> list[LoglikelihoodSingleTokenResponse]:
        raise NotImplementedError

    def tok_encode_jax(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None) -> TokenSequence:
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        if isinstance(str_to_encode, str):
            return self.tokenizer.encode(str_to_encode, add_special_tokens=add_special_tokens)
        return self.tokenizer(
            str_to_encode,
            padding=True,
            add_special_tokens=add_special_tokens,
            return_tensors="np",
        )

    def tok_decode_jax(self, tokens: jax.Array) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)



def main():
    tasks = ["lighteval|arc:easy|0|0",
             "leaderboard|arc:challenge|0|0",
             "helm|piqa|0|0",
             "helm|siqa|0|0",
             "leaderboard|hellaswag|0|0",
             "helm|openbookqa|0|0",
             "leaderboard|winogrande|0|0",
             "lighteval|triviaqa|0|0",
             "lighteval|race:high|0|0"]
    tasks = "leaderboard|hellaswag|0|0"

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
        output_dir=str(output_dir),
        save_details=False,
    )

    params = PipelineParameters(
        launcher_type=ParallelismManager.CUSTOM,
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=params,
        evaluation_tracker=tracker,
        model_config=model_cfg,
    )

    pipeline.evaluate()  # run the eval
    # pipeline.save_and_push_results()  # write files (and push if enabled)
    pipeline.show_results()


if __name__ == "__main__":
    main()
