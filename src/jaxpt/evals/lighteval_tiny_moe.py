import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./alpha-448101-282bc1b884cd.json"

from time import time

from dataclasses import dataclass

from tqdm import tqdm 

from typing import Union, Optional
import argparse
from pathlib import Path
import logging
import warnings

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import numpy as np

from jaxpt.models.tiny_moe import Tiny_MoE, Tiny_MoE_Config
from jaxpt.checkpointers import load_checkpoint_from_gcloud
from jaxpt.utils import count_params

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_output import (
    Batch,
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset, LoglikelihoodSingleTokenDataset
from lighteval.tasks.requests import (
    Request,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)

logger = logging.getLogger(__name__)

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
spec = jax.sharding.PartitionSpec("devices",)
sharding = jax.sharding.NamedSharding(mesh, spec)

STARTING_BATCH_SIZE = 512

@dataclass
class Jaxpt_Batch:
    input_ids: np.ndarray
    input_mask: np.ndarray
    input_lengths: list[int]
    truncated: list[int]
    padded: list[int]


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
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
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

    @nnx.jit(static_argnums=(0, ))
    def _call_model(self, x):
        return self.model(x)

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
        res = []

        split_id = 0
        print("num dataset splits", len(dataset))
        for split in tqdm(dataset.splits_iterator(), disable=self.disable_tqdm):
            print('starting split', split_id)
            split_id += 1
            context_enc = split[0].tokenized_context
            continuation_enc = split[0].tokenized_continuation
            max_input_length = max(min(self.max_length, len(context_enc + continuation_enc) - 1), 1)


            dataloader = DataLoader(split, batch_size=64, collate_fn=lambda batch: batch)

            print("num batches", len(dataloader))
            batch_id = 0

            for batch in tqdm(dataloader, disable=self.disable_tqdm):
                prepared_batch = self.prepare_batch_logprob(
                    batch,
                    padding_length=max_input_length,
                    max_context=max_input_length,
                )
                print("starting batch", batch_id)
                batch_id += 1 

                x = prepared_batch.input_ids
                if x.shape[0] % num_devices == 0:
                    x = jnp.array(prepared_batch.input_ids, device=sharding)
                else:
                    x = jnp.array(prepared_batch.input_ids)

                with mesh: 
                    start = time()
                    model_output = self._call_model(x)
                    logits = jax.nn.log_softmax(model_output, axis=-1)  # [batch, padding_length, vocab]
                    duration = time() - start 
                    print("model time", duration)
            
                    logits_sum = []
                    max_equals = []
                    batch_cont_tokens = []
                    for cur_request, cur_logits, inplen in zip(batch, logits, prepared_batch.input_lengths):
                        start = time()
                        cont_toks = jnp.array(cur_request.tokenized_continuation, dtype=jnp.int32)
                        contlen = cont_toks.shape[0]
                        # We only look at the continuation tokens
                        if contlen > inplen:
                            # Continuation is longer than the input size, we are in rolling mode (only continuation)
                            cur_logits = jnp.expand_dims(cur_logits, 0)  # [1, seq, vocab]
                            cont_toks = jnp.expand_dims(cont_toks[:inplen], 0)  # [1, seq]
                        else:
                            cur_logits = (
                                jnp.expand_dims(cur_logits[inplen - contlen : inplen], 0)
                            )  # [1, seq, voc]
                            cont_toks = jnp.expand_dims(cont_toks, 0)

                        # Check if per-token argmax is exactly equal to continuation
                        greedy_tokens = cur_logits.argmax(axis=-1)
                        # Sometimes the continuation is longer than allowed by the model, we only look at the first tokens
                        max_equal = (greedy_tokens == cont_toks).all() #.squeeze(0)

                        # Obtain log-probs at the corresponding continuation token indices
                        # torch.gather(cur_logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                        cur_logits = jnp.take_along_axis(
                            cur_logits, jnp.expand_dims(cont_toks, axis=-1), axis=2
                        ).squeeze(-1)  # [1, seq]

                        # Answer: (log prob, is-exact-match)
                        logits_sum.append(cur_logits.sum())
                        max_equals.append(max_equal)
                        batch_cont_tokens.append(cont_toks)

                    # Sync all
                    # Need reshaping before gather
                    batched_inputs, len_inputs = self.pad_and_gather(prepared_batch.input_ids)
                    max_cont_tokens_length = max(len(c[0]) for c in batch_cont_tokens)
                    # These are the true lengths of the continuation tokens, we have to save them to be able to removed padding tokens from the generated tokens.
                    batch_cont_token_lengths = jnp.array([c.shape[1] for c in batch_cont_tokens])
                    batch_cont_tokens = jnp.concatenate(
                        [
                            jnp.pad(
                                c,
                                ((0, 0), (0, max_cont_tokens_length - c.shape[1])),
                                constant_values=self.tokenizer.pad_token_id
                            )
                            for c in batch_cont_tokens
                        ],
                        axis=0,
                    )
                    batch_cont_tokens, _ = self.pad_and_gather(batch_cont_tokens)
                    # Can be gathered as such
                    logits = jnp.array(logits_sum)
                    max_equal = jnp.array(max_equals)
                    batch_truncated = jnp.array(prepared_batch.truncated)
                    batch_padded = jnp.array(prepared_batch.padded)
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
                        input_tokens = batched_input[:len_input].tolist()
                        generated_tokens = cont_tokens[:len_token].tolist()

                        answer = LoglikelihoodResponse(
                            # todo: we might want to store the logits unsummed
                            result=(float(logit.sum()), bool(maxe)),
                            input_tokens=input_tokens,
                            generated_tokens=generated_tokens,
                            truncated_tokens_count=int(trunc),
                            padded_tokens_count=int(padded),
                        )
                        res.append(answer)
                    duration = time() - start 
                    print("loop time", duration)

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

    def prepare_batch_logprob(
        self, batch: list[Request], padding_length: int, max_context: Optional[int] = None, single_token: bool = False
    ):
        """Tokenize a batch of inputs and return also the length, truncations and padding.
        This step is done manually since we tokenize log probability inputs together with their continuation,
        to manage possible extra spaces added at the start by tokenizers, see tok_encode_pair.
        """
        if single_token:
            inputs = [request.tokenized_context for request in batch]
        else:
            inputs = [
                request.tokenized_context + request.tokenized_continuation[:-1] for request in batch
            ]  # The last token (an eos) doesn't need to be given to the model

        input_tokens = []
        attention_masks = []
        input_lengths = []
        truncated = []
        padded = []

        if max_context is None:
            logger.warning("max_context is None, using max_length")
            max_context = self.max_length

        # Each sample is concatenated and cut to length or padded to max_length
        for orig_tokens in inputs:
            truncated.append(max(len(orig_tokens) - max_context, 0))

            # Truncate from the left if needed to fit in the model's context
            tokens = np.array((orig_tokens)[-max_context:], dtype=np.int32)
            sequence_len = tokens.shape[0]

            # We add padding, if needed
            pad_len = padding_length if padding_length is not None else sequence_len

            if pad_len - sequence_len < 0:
                logger.warning(f"Padding length {pad_len} is smaller than input length {sequence_len}")
                raise ValueError("Negative padding")

            padded.append(pad_len - sequence_len)
            # Right padding, since we ignore these logprobs in the end
            tokens = np.pad(tokens, (0, pad_len - sequence_len), constant_values=self.tokenizer.pad_token_id)

            # We create the attention mask to ignore padding
            mask = (tokens == self.tokenizer.pad_token_id)
            attention_masks.append(mask[None, ...])

            input_tokens.append(tokens[None, ...])  # [1, padding_length]
            input_lengths.append(sequence_len)

        batched_inputs = np.concatenate(input_tokens, axis=0)  # [batch, padding_length]
        attention_masks = np.concatenate(attention_masks, axis=0)

        return Jaxpt_Batch(
            input_ids=batched_inputs,
            input_mask=attention_masks,
            input_lengths=input_lengths,
            truncated=truncated,
            padded=padded,
        )

    def pad_and_gather(
        self, output_tensor
    ):
        """
        Pads the `output_tensor` to the maximum length and gathers the lengths across processes.

        Returns:
            Tuple: The padded output tensor and the gathered length tensor.
        """
        length_tensor = jnp.array([output_tensor.shape[-1]] * output_tensor.shape[0])
        max_length = int(length_tensor.max())
        pad_width = [(0, 0)] * output_tensor.ndim
        pad_width[-1] = (0, max_length - output_tensor.shape[-1])
        output_tensor = jnp.pad(output_tensor, pad_width, constant_values=self.tokenizer.pad_token_id)
        return output_tensor, length_tensor


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
