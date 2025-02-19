from dataclasses import dataclass
import jax
import flax.nnx as nnx


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384


class MLP(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embed, 4 * config.n_embed, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embed, config.n_embed, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = flax.nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x


class Block(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embed, rngs=rngs)
        #self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embed, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)
    
    def __call__(self, x):
        res = x
        x = self.ln_1(x)    
        x = self.attn(x)
        x = x + res

        res = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + res
        return x


class Transformer(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.wte = nnx.Embed(config.vocab_size, config.n_embed, rngs=rngs)
        self.wpe = nnx.Embed(config.block_size, config.n_embed, rngs=rngs)  
        #self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(config.n_embed, rngs=rngs)


class GPT2(nnx.Module):

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.transformer = Transformer(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.n_embed, config.vocab_size, rngs=rngs, use_bias=False)



    

