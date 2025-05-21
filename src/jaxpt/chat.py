import jax
import jax.numpy as jnp
from flax import nnx

from jaxpt.infer import generate_completions, generate, generate_chat
from jaxpt.models import Mobile_LLM, MobileLLM_Config
from jaxpt.models.mobile_llm import from_hf_pretrained
from jaxpt.utils import count_params

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
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)

device = "cpu"

key = jax.random.PRNGKey(1337)
rngs = nnx.Rngs(key)
config = MobileLLM_Config(dtype=jnp.bfloat16, \
                    vocab_size=49152,
                    n_embed=576,
                    n_head=9,
                    n_kv_head=3,
                    n_mlp_hidden=1536,
                    use_cache=False,
                    sdpa_implementation="cudnn" if device=="gpu" else "xla")
#nnx.display(config)
#m = Mobile_LLM(config, rngs)
#graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
#nnx.display(state)
#m = load_checkpoint("run_20250311_uqdwjq", 5600)
m = from_hf_pretrained(config, rngs)

graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
total_params = count_params(m)

logger.info(f"Parameter Count: {total_params:,}")
#nnx.display(state)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

while True:
    try:
        user_input = input("You: ")
        logger.debug(f"User input received: {user_input}")
    except (EOFError, KeyboardInterrupt):
        logger.info("Chat session ended by user.")
        break

    generate_chat(m, enc=tokenizer, format="chatml", question=user_input,
                  logger=logger)