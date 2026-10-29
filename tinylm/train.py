"""Train one of the four models, resuming automatically from OUT/ckpt.msgpack.

python -m tinylm.train baseline --data ~/project/data --out ~/project/ckpt/baseline
python -m tinylm.train mdlm --tiny --overfit --data data --out ckpt/tiny   # debug: loss -> ~0
"""
import argparse, json, os, time
import jax, jax.numpy as jnp, optax
from flax import serialization

from tinylm import configs
from tinylm.configs import MASK
from tinylm.data import batches
from tinylm.model import LM, n_params


def ar_loss(model, params, batch, key):
    logits = model.apply(params, batch[:, :-1])
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:]).mean()


def mdlm_loss(model, params, batch, key, eps=1e-3):
    """MDLM objective: mask each token with prob t ~ U(eps, 1), predict the masked tokens,
    weight by 1/t. Its mean is an upper bound on NLL per token, so it compares to ar_loss."""
    x0 = batch[:, :-1]
    kt, km = jax.random.split(key)
    t = jax.random.uniform(kt, (x0.shape[0], 1), minval=eps)
    m = jax.random.uniform(km, x0.shape) < t
    logits = model.apply(params, jnp.where(m, MASK, x0)).at[..., MASK].set(-1e9)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, x0)
    return (ce * m / t).mean()


LOSS = {"ar": ar_loss, "mdlm": mdlm_loss}


def optimizer(c, steps):
    warmup = min(c["warmup"], steps // 10)  # short debug runs still get a decay phase
    lr = optax.warmup_cosine_decay_schedule(0.0, c["lr"], warmup, steps, c["min_lr"])
    matrices = lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)  # no decay on norms
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.adamw(lr, b1=0.9, b2=0.95, weight_decay=c["weight_decay"], mask=matrices))


def save(path, obj):
    with open(path + ".tmp", "wb") as f:
        f.write(serialization.to_bytes(obj))
    os.replace(path + ".tmp", path)  # atomic: a session killed mid-write keeps the old file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=configs.CONFIGS)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--overfit", action="store_true", help="train on one fixed batch")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--ckpt_minutes", type=float, default=10)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=1000)
    a = ap.parse_args()

    c = configs.get(a.name, a.tiny)
    B, T = c["batch_size"], c["block_size"]
    steps = a.steps or c["max_tokens"] // (B * T)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**c, "name": a.name, "steps": steps}, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    model, loss_fn, opt = LM(c), LOSS[c["objective"]], optimizer(c, steps)
    params = model.init(jax.random.PRNGKey(c["seed"]), jnp.zeros((1, T), jnp.int32))
    state = {"params": params, "opt": opt.init(params), "step": 0}
    ckpt = os.path.join(a.out, "ckpt.msgpack")
    if os.path.exists(ckpt):
        state = serialization.from_bytes(state, open(ckpt, "rb").read())
        print(f"resumed at step {state['step']}")
    start = int(state["step"])
    print(f"{a.name}: {n_params(state['params'])/1e6:.1f}M params (incl. embedding), "
          f"{configs.n_params(c)/1e6:.1f}M non-embedding, {steps} steps of {B*T} tokens")

    @jax.jit
    def train_step(params, opt_state, batch, key):
        loss, g = jax.value_and_grad(lambda p: loss_fn(model, p, batch, key))(params)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    eval_loss = jax.jit(lambda p, batch, key: loss_fn(model, p, batch, key))
    val_iter = batches(os.path.join(a.data, "validation.bin"), B, T, seed=1)
    val = [next(val_iter) for _ in range(20)]  # same 20 batches every time

    def validate(p):
        return float(sum(eval_loss(p, v, jax.random.PRNGKey(i)) for i, v in enumerate(val)) / len(val))

    train_bin = os.path.join(a.data, "train.bin")
    train_iter = batches(train_bin, B, T, seed=c["seed"] + start)
    fixed = next(batches(train_bin, B, T, seed=c["seed"])) if a.overfit else None
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    params, opt_state = state["params"], state["opt"]
    t0 = last_ckpt = time.time()
    for step in range(start, steps):
        batch = fixed if a.overfit else next(train_iter)
        params, opt_state, loss = train_step(params, opt_state, batch, jax.random.PRNGKey(step))
        done = step + 1 == steps
        if step % a.log_every == 0 or done:
            rec = {"step": step, "tokens": (step + 1) * B * T, "loss": float(loss), "sec": round(time.time() - t0, 1)}
            if step % a.eval_every == 0 or done:
                rec["val"] = validate(params)
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if done or time.time() - last_ckpt > a.ckpt_minutes * 60:
            save(ckpt, {"params": params, "opt": opt_state, "step": step + 1})
            last_ckpt = time.time()
    save(os.path.join(a.out, "params.msgpack"), params)  # weights only, for eval and the demo


if __name__ == "__main__":
    main()
