"""One model file for all four models: LLaMA-style blocks (RoPE, RMSNorm, SwiGLU).

c["loop"]      how many times the stack of blocks is applied (1 = plain transformer)
c["objective"] "ar" uses a causal mask, "mdlm" attends both ways
"""
import jax, jax.numpy as jnp
import flax.linen as nn

init = nn.initializers.normal(0.02)


def rope(x):
    """Rotary position embedding on (B, T, heads, head_dim)."""
    T, h = x.shape[1], x.shape[-1] // 2
    freq = 1.0 / (10000 ** (jnp.arange(h) / h))
    ang = jnp.arange(T)[:, None] * freq[None]
    cos, sin = jnp.cos(ang)[None, :, None], jnp.sin(ang)[None, :, None]
    x1, x2 = x[..., :h], x[..., h:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).astype(x.dtype)


class Block(nn.Module):
    c: dict

    @nn.compact
    def __call__(self, x, causal):
        c, dt = self.c, jnp.dtype(self.c["dtype"])
        B, T, d = x.shape
        H = c["n_head"]
        out_init = nn.initializers.normal(0.02 / (2 * c["n_layer"] * c["loop"]) ** 0.5)

        h = nn.RMSNorm(dtype=dt)(x)
        qkv = nn.Dense(3 * d, use_bias=False, kernel_init=init, dtype=dt)(h).reshape(B, T, 3, H, d // H)
        q, k, v = rope(qkv[:, :, 0]), rope(qkv[:, :, 1]), qkv[:, :, 2]
        a = jax.nn.dot_product_attention(q, k, v, is_causal=causal).reshape(B, T, d)
        x = x + nn.Dense(d, use_bias=False, kernel_init=out_init, dtype=dt)(a)

        h = nn.RMSNorm(dtype=dt)(x)
        g = nn.Dense(2 * c["ffn"], use_bias=False, kernel_init=init, dtype=dt)(h)
        g, u = jnp.split(g, 2, -1)
        return x + nn.Dense(d, use_bias=False, kernel_init=out_init, dtype=dt)(nn.silu(g) * u)


class LM(nn.Module):
    c: dict

    @nn.compact
    def __call__(self, idx, loop=None):
        """idx (B, T) int -> logits (B, T, vocab) float32. `loop` overrides c["loop"] at test time."""
        c, dt = self.c, jnp.dtype(self.c["dtype"])
        emb = nn.Embed(c["vocab"], c["d"], embedding_init=init, dtype=dt)
        blocks = [Block(c) for _ in range(c["n_layer"])]
        x = emb(idx)
        for _ in range(loop or c["loop"]):
            for b in blocks:
                x = b(x, causal=c["objective"] == "ar")
        x = nn.RMSNorm(dtype=dt)(x)
        return emb.attend(x).astype(jnp.float32)  # output head tied to the embedding


def n_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
