"""Figs 9-11 of TinyStories: greedy completions of the paper's 24 prompts, one column per model.
Writes results/figs9_11.csv with an empty `label` column per cell (success / partial / failure)
to fill in by hand, as the paper did.

python -m eval.figs9_11 --ckpt ~/project/ckpt
"""
import argparse, csv, json, os

from eval.models import available, load

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="dir with our trained models; TinyStories checkpoints always run")
    ap.add_argument("--out", default="results/figs9_11.csv")
    ap.add_argument("--max_new", type=int, default=40)
    a = ap.parse_args()

    prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts_fig9_11.json")))
    rows = []
    for name in available(a.ckpt):
        m = load(name, a.ckpt)
        for fig, ps in prompts.items():
            for i, p in enumerate(ps):
                rows.append({"figure": fig, "prompt_id": i, "model": name, "prompt": p,
                             "completion": m.generate(p, max_new=a.max_new), "label": ""})
        print(name, "done", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} completions to {a.out}")
