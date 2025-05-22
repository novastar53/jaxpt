from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from jaxpt.infer import generate_completions, generate, generate_chat
from jaxpt.models import Mobile_LLM, MobileLLM_Config
from jaxpt.models.mobile_llm import from_hf_pretrained
from jaxpt.utils import count_params
from jaxpt.checkpointers import load_checkpoint

from transformers import AutoTokenizer

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Prevent logs from being propagated to the root logger
logger.propagate = False
# Remove all existing handlers
logger.handlers.clear()
# Create a console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)

device = "cpu"

key = jax.random.PRNGKey(1337)
rngs = nnx.Rngs(key)
config = MobileLLM_Config(
    name="mobile_llm_sft_webinstruct",
    dtype=jnp.bfloat16,
    vocab_size=49152,
    n_embed=576,
    n_head=9,
    n_kv_head=3,
    n_mlp_hidden=1536,
    use_cache=False,
    sdpa_implementation="cudnn" if device == "gpu" else "xla",
)
# nnx.display(config)
# m = Mobile_LLM(config, rngs)
# graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
# nnx.display(state)
output_dir = Path().absolute() / "alpha_training_runs"
m = load_checkpoint(
    Mobile_LLM, output_dir, config, "run_20250521_pakugd", 9000, rngs
)
# m = from_hf_pretrained(config, rngs)

graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
total_params = count_params(m)

logger.info(f"Parameter Count: {total_params:,}")
# nnx.display(state)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

try:
    user_input = input("You: ")
    logger.debug(f"User input received: {user_input}")
    x = generate_chat(
        m, enc=tokenizer, format="chatml", question=user_input, logger=logger
    )

    while True:
        user_input = input("You: ")
        x = generate_chat(
            m,
            x_prev=x,
            enc=tokenizer,
            format="chatml",
            question=user_input,
            logger=logger,
        )
except (EOFError, KeyboardInterrupt):
    logger.info("Chat session ended by user.")
