"""Run lm-evaluation-harness tasks on one of our JAX models, so it sees exactly the
prompts the TinyStories checkpoints see.

python -m eval.harness_lm ~/project/ckpt/baseline --tasks mmlu,blimp --out results/harness/baseline
python -m eval.harness_lm ~/project/ckpt/looped --loop 4 --out results/harness/looped-loop4
"""
import argparse, json, os
from lm_eval import simple_evaluate
from lm_eval.api.model import LM as HarnessLM
from tqdm import tqdm

from tinylm.configs import EOS
from tinylm.sample import Model


class JaxLM(HarnessLM):
    def __init__(self, path, loop=None, n_mc=32):
        super().__init__()
        self.m, self.n_mc = Model(path, loop, dtype="float32"), n_mc  # fp32, like the HF baselines

    def loglikelihood(self, requests, disable_tqdm=False):
        pairs = []
        for req in requests:
            ctx, cont = req.args
            # like the harness's HF model: trailing spaces of the context belong to the continuation
            k = len(ctx) - len(ctx.rstrip())
            if k:
                ctx, cont = ctx[:-k], ctx[-k:] + cont
            enc = self.m.enc.encode_ordinary
            if ctx:  # like HFLM: encode context+continuation together, then split
                ctx_ids = enc(ctx)
                cont_ids = enc(ctx + cont)[len(ctx_ids):]
            else:
                ctx_ids, cont_ids = [EOS], enc(cont)
            pairs.append((ctx_ids, cont_ids))
        # is_greedy only feeds LAMBADA's accuracy; skipping it elsewhere saves the diffusion
        # models' step-by-step unmasking (one TPU call per continuation token)
        greedy = [req.task_name.startswith("lambada") for req in requests]
        out, step = [], 4096
        for s in tqdm(range(0, len(pairs), step), disable=disable_tqdm):
            out += self.m.loglik_batch(pairs[s:s + step], n_mc=self.n_mc, need_greedy=greedy[s:s + step])
        return out

    # ponytail: none of our six tasks generate text or score whole documents
    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tasks", default="mmlu,blimp,hellaswag,piqa,arc_easy,lambada_openai")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loop", type=int)
    ap.add_argument("--n_mc", type=int, default=32, help="Monte-Carlo samples per answer (diffusion)")
    ap.add_argument("--limit", type=int, help="examples per task, for a quick check")
    a = ap.parse_args()
    res = simple_evaluate(model=JaxLM(a.path, a.loop, a.n_mc), tasks=a.tasks.split(","), limit=a.limit)
    os.makedirs(a.out, exist_ok=True)
    res.pop("samples", None)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    for task, r in res["results"].items():
        print(task, {k: v for k, v in r.items() if k.startswith("acc") and "stderr" not in k})
