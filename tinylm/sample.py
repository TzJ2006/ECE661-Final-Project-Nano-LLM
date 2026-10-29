"""Load a trained model, write stories, and score continuations (used by eval and the demo).

python -m tinylm.sample ~/project/ckpt/baseline "Once upon a time"
python -m tinylm.sample ~/project/ckpt/mdlm "Once upon a time" --steps 64
"""
import argparse, json, os
import numpy as np, jax, jax.numpy as jnp
from flax import serialization

from tinylm.configs import EOS, MASK
from tinylm.data import encoder
from tinylm.model import LM


class Model:
    def __init__(self, out, loop=None, dtype=None):
        """out: a training output dir. loop: test-time loop count (looped models only)."""
        self.c = json.load(open(os.path.join(out, "config.json")))
        if dtype:
            self.c["dtype"] = dtype
        self.lm, self.loop = LM(self.c), loop
        template = self.lm.init(jax.random.PRNGKey(0), jnp.zeros((1, 8), jnp.int32))
        self.params = serialization.from_bytes(template, open(os.path.join(out, "params.msgpack"), "rb").read())
        self.T = self.c["block_size"]
        self.ar = self.c["objective"] == "ar"
        self.enc = encoder()
        self._fwd = jax.jit(lambda p, x: self.lm.apply(p, x, loop=self.loop))

        def tok_scores(p, x, y):
            """Per position: log p(y), argmax, and its log-prob. Gathered on device so the
            (batch, T, vocab) log-probs never leave the TPU."""
            lp = jax.nn.log_softmax(self.lm.apply(p, x, loop=self.loop)[..., :MASK])
            return jnp.take_along_axis(lp, y[..., None], -1)[..., 0], lp.argmax(-1), lp.max(-1)
        self._scores = jax.jit(tok_scores)

    def logits(self, x):
        return self._fwd(self.params, jnp.asarray(x, jnp.int32))

    def scores(self, x, y):
        return [np.asarray(a) for a in self._scores(self.params, jnp.asarray(x, jnp.int32), jnp.asarray(y, jnp.int32))]

    def _mask_pad(self, ids):
        """Diffusion models only ever saw full T-token windows, so short inputs are padded to T
        with MASK: the model then treats the rest as unknown text (exact marginal), one jit shape."""
        x = np.full(self.T, MASK, np.int32)
        x[:len(ids)] = ids
        return x

    # ---------- generation ----------
    def generate(self, prompt, max_new=200, temperature=0.0, steps=64, seed=0):
        ids = self.enc.encode_ordinary(prompt)[-(self.T - max_new):]
        gen = self._gen_ar if self.ar else self._gen_mdlm
        out = gen(ids, max_new, temperature, steps, np.random.default_rng(seed))
        out = out[:out.index(EOS)] if EOS in out else out
        return self.enc.decode(out)

    def _pick(self, logits, temperature, rng):
        if temperature == 0:
            return int(np.argmax(logits))
        p = np.exp((logits - logits.max()) / temperature)
        return int(rng.choice(len(p), p=p / p.sum()))

    def _gen_ar(self, ids, max_new, temperature, steps, rng):
        # ponytail: recomputes the full window each token (no KV cache); fine at 33M / 512 tokens
        buf, n = np.full((1, self.T), EOS, np.int32), len(ids)
        buf[0, :n] = ids
        for i in range(n, n + max_new):
            logits = np.asarray(self.logits(buf)[0, i - 1, :MASK])
            buf[0, i] = self._pick(logits, temperature, rng)
            if buf[0, i] == EOS:
                break
        return buf[0, n:i + 1].tolist()

    def _gen_mdlm(self, ids, max_new, temperature, steps, rng):
        """Start with max_new masks after the prompt; each step commits the most confident
        predictions (LLaDA's low-confidence remasking) so all are filled after `steps` steps."""
        n = len(ids)
        x = self._mask_pad(ids)[None]  # positions past n + max_new stay MASK and are never filled
        for s in range(steps):
            masked = n + np.where(x[0, n:n + max_new] == MASK)[0]
            if len(masked) == 0:
                break
            logits = np.asarray(self.logits(x)[0, masked, :MASK])
            pred = np.array([self._pick(l, temperature, rng) for l in logits])
            prob = np.exp(logits - logits.max(-1, keepdims=True))
            conf = (prob / prob.sum(-1, keepdims=True))[np.arange(len(pred)), pred]
            k = int(np.ceil(len(masked) / (steps - s)))
            keep = np.argsort(-conf)[:k]
            x[0, masked[keep]] = pred[keep]
        return x[0, n:n + max_new].tolist()

    # ---------- scoring (for lm-evaluation-harness) ----------
    def loglik(self, ctx, cont, n_mc=32, seed=0):
        return self.loglik_batch([(ctx, cont)], n_mc, seed)[0]

    def loglik_batch(self, pairs, n_mc=32, seed=0, bs=64, need_greedy=None):
        """[(ctx, cont)] token lists -> [(log p(cont | ctx), is_greedy)]; ctx must be non-empty.
        AR: exact, teacher-forced greedy check. MDLM: Monte-Carlo ELBO estimate (LLaDA eq. 6):
        mask l of the L continuation tokens, weight the masked log-probs by L/l, with l spread
        evenly over 1..L across samples (low variance). MDLM is_greedy follows LLaDA: unmask the
        continuation one most-confident token at a time, then compare with the target."""
        items = []
        for ctx, cont in pairs:
            ids = (ctx + cont)[-self.T:]
            items.append((ids, len(ids), min(len(cont), len(ids) - 1)))
        return self._ar_batch(items, bs) if self.ar else self._mdlm_batch(items, n_mc, seed, bs, need_greedy)

    def _ar_batch(self, items, bs):
        """bs requests per call, grouped by length; widths are multiples of 64 (few jit shapes)."""
        out = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: items[i][1])
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            top = max(items[i][1] for i in idx)
            x = np.full((bs, min(top + -top % 64, self.T)), EOS, np.int32)
            for r, i in enumerate(idx):
                x[r, :items[i][1]] = items[i][0]
            lp, am, _ = self.scores(x, np.roll(x, -1, 1))
            for r, i in enumerate(idx):
                _, n, L = items[i]
                pos = np.arange(n - L - 1, n - 1)
                out[i] = (float(lp[r, pos].sum()), bool((am[r, pos] == x[r, pos + 1]).all()))
        return out

    def _mdlm_batch(self, items, n_mc, seed, bs, need_greedy):
        """ELBO: every (request, sample) masked copy is one row, packed bs rows per call. A
        1-token continuation needs a single row (masking it is the only option, so it is exact).
        Greedy unmasking (bs requests in parallel) runs only where need_greedy is set."""
        u = np.random.default_rng(seed).random()
        perm = np.random.default_rng(seed + 1)
        n_rows = lambda L: 1 if L == 1 else n_mc
        specs = [(i, j) for i, (_, _, L) in enumerate(items) for j in range(n_rows(L))]
        ll, cnt = np.zeros(len(items)), np.zeros(len(items))
        for s in range(0, len(specs), bs):
            X = np.full((bs, self.T), MASK, np.int32)
            Y = np.zeros((bs, self.T), np.int32)  # targets; unweighted positions ignored
            W = np.zeros((bs, self.T), np.float32)
            for k, (i, j) in enumerate(specs[s:s + bs]):
                ids, n, L = items[i]
                l = int(((u + j / n_rows(L)) % 1) * L) + 1
                m = n - L + perm.choice(L, l, replace=False)
                X[k, :n] = ids; X[k, m] = MASK
                Y[k, :n] = ids; W[k, m] = L / l
            lp = self.scores(X, Y)[0]
            for k, (i, j) in enumerate(specs[s:s + bs]):
                ll[i] += (lp[k] * W[k]).sum(); cnt[i] += 1

        greedy = [False] * len(items)
        need = [i for i in range(len(items)) if need_greedy is None or need_greedy[i]]
        for s in range(0, len(need), bs):
            idx = need[s:s + bs]
            x = np.full((bs, self.T), MASK, np.int32)  # spare rows stay all-MASK and are ignored
            for r, i in enumerate(idx):
                ids, n, L = items[i]
                x[r, :n - L] = ids[:n - L]
            for _ in range(max(items[i][2] for i in idx)):
                _, am, conf = self.scores(x, np.zeros_like(x))
                for r, i in enumerate(idx):
                    ids, n, L = items[i]
                    masked = n - L + np.where(x[r, n - L:n] == MASK)[0]
                    if len(masked):
                        best = masked[np.argmax(conf[r, masked])]
                        x[r, best] = am[r, best]
            for r, i in enumerate(idx):
                ids, n, L = items[i]
                greedy[i] = bool((x[r, n - L:n] == ids[n - L:]).all())
        return [(float(a / c), g) for a, c, g in zip(ll, cnt, greedy)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=64, help="diffusion unmasking steps")
    ap.add_argument("--loop", type=int)
    a = ap.parse_args()
    m = Model(a.out, a.loop)
    print(a.prompt + m.generate(a.prompt, temperature=a.temperature, steps=a.steps))
