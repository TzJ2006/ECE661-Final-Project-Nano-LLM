"""The four models. Each is the baseline dict with one or two fields changed.

Baseline blocks match TinyStories-33M (GPT-Neo: d=768, 4 layers, FFN 3072):
a SwiGLU FFN of width 2048 has the same weight count as a GELU FFN of width 3072.
"""

VOCAB = 50257          # GPT-2/GPT-Neo BPE, same tokenizer as the TinyStories checkpoints
EOS = 50256
MASK = 50257           # extra id used only by the diffusion models

BASE = dict(
    vocab=VOCAB + 1,   # + MASK, so all four models share one embedding size
    d=768, n_layer=4, n_head=12, ffn=2048,
    block_size=512,
    loop=1,            # times the stack of n_layer blocks is applied
    objective="ar",    # "ar" (left to right) or "mdlm" (masked diffusion)
    # training recipe, identical for every model
    batch_size=64, lr=6e-4, min_lr=6e-5, warmup=1000, weight_decay=0.1,
    max_tokens=500_000_000, seed=0,
)

# ponytail: looping keeps the same blocks and width, so parameters stay equal
# and only FLOPs/token grow; swap to fewer, wider shared blocks if the report needs that axis.
CONFIGS = {
    "baseline": BASE,
    "looped": {**BASE, "loop": 2},
    "mdlm": {**BASE, "objective": "mdlm"},
    "looped_mdlm": {**BASE, "loop": 2, "objective": "mdlm"},
}

# debug overrides for overfitting one batch on CPU/TPU in seconds
TINY = dict(d=64, n_layer=2, n_head=4, ffn=128, block_size=64, batch_size=8, max_tokens=200_000, warmup=10)


def get(name, tiny=False):
    return {**CONFIGS[name], **(TINY if tiny else {})}


def n_params(c):
    """Non-embedding parameter count (attention + FFN + norms), the usual way to report LM size."""
    d = c["d"]
    per_block = 4 * d * d + 3 * d * c["ffn"] + 2 * d
    return c["n_layer"] * per_block + d


def flops_per_token(c):
    """Forward+backward FLOPs per token, 6N with the looped compute counted (Kaplan et al.)."""
    attn = 12 * c["n_layer"] * c["d"] * c["block_size"]  # attention-score term, small at this size
    return 6 * (n_params(c) - c["d"]) * c["loop"] + 6 * c["d"] * c["vocab"] + attn * c["loop"]


if __name__ == "__main__":
    for k in CONFIGS:
        c = get(k)
        print(f"{k:12s} params={n_params(c)/1e6:5.1f}M  flops/token={flops_per_token(c)/1e6:6.0f}M")
