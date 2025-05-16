import jax
import jax.numpy as jnp
from flax import nnx

from jaxpt.infer import generate_completions, generate, generate_chat
from jaxpt.models import Mobile_LLM, MobileLLM_Config
from jaxpt.models.mobile_llm import from_hf_pretrained
from jaxpt.utils import count_params

from transformers import AutoTokenizer

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

print(f"Parameter Count: {total_params:,}")
#nnx.display(state)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

while True:
    try:
        user_input = input("You: ")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    generate_chat(m, enc=tokenizer, prefix=user_input)