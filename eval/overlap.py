"""TinyStories Sec 4.4.1 (Figs 14-17): are fluent completions copied from the training data?

100 random training stories are cut after 40% of their words; each model completes the prefix
greedily. With word k-gram sets G_k and precision R_kp(A, B) = |G_k(A) & G_k(B)| / |G_k(A)|:
  Fig 14  s_i = R_2p(T_i, T'_i)              completion vs the story's real ending
  Fig 15  r_i = max_j!=i R_2F(T_i, T_j)      completion vs the model's other completions (F-measure)
  Fig 16  log10 of each completion k-gram's frequency in the training corpus, k = 2..5 (-10 = never)
  Fig 17  h_i = max_j R_kp(T_i, S_j), k=2,3  completion vs the closest training story
"original" = the real endings T'_i (for Fig 17 their own source story is excluded).
ponytail: k-grams are counted as sets of lowercase words; the paper does not state its choice.

python -m eval.overlap --ckpt ~/project/ckpt --out results/overlap.json
"""
import argparse, json, os, re
from multiprocessing import Pool
import numpy as np
from scipy import sparse

from eval.models import available, load

WORD = re.compile(r"[a-z0-9']+")
words = lambda s: WORD.findall(s.lower())
grams = lambda w, k: {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def prec(a, b):
    return len(a & b) / len(a) if a else 0.0


def f1(a, b):
    p, q = prec(a, b), prec(b, a)
    return 2 * p * q / (p + q) if p + q else 0.0


def cut(text, frac=0.4):
    spans = [m.end() for m in WORD.finditer(text.lower())]
    return text[:spans[int(len(spans) * frac) - 1]], text[spans[int(len(spans) * frac) - 1]:]


# ---- one pass over the corpus per k, split over all CPUs ----
_Q = None


def _init(q):
    global _Q
    _Q = q


def _scan(args):
    """For a chunk of stories: k-gram occurrence counts per query column, and the
    stories x columns presence matrix (for the closest-story search)."""
    texts, k = args
    vocab = _Q[k]
    counts = np.zeros(len(vocab), np.int64)
    rows, cols, total = [], [], 0
    for r, t in enumerate(texts):
        w = words(t)
        total += max(0, len(w) - k + 1)
        hit = [vocab[g] for g in (tuple(w[i:i + k]) for i in range(len(w) - k + 1)) if g in vocab]
        np.add.at(counts, hit, 1)
        u = set(hit)
        rows += [r] * len(u); cols += list(u)
    pres = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
    return counts, total, pres


def corpus_stats(corpus, comps, src_idx, ks=(2, 3, 4, 5), closest_ks=(2, 3), chunk=20_000):
    """comps: {model: [completion word lists]}. Returns per-k log10 frequencies and closest-story precision."""
    flat = [(m, i, w) for m, ws in comps.items() for i, w in enumerate(ws)]
    q = {k: {} for k in ks}
    sets = {k: [grams(w, k) for _, _, w in flat] for k in ks}
    for k in ks:
        for s in sets[k]:
            for g in s:
                q[k].setdefault(g, len(q[k]))
    out = {"freq": {}, "closest": {}}
    with Pool(os.cpu_count() - 2, initializer=_init, initargs=(q,)) as pool:
        for k in ks:
            counts, total = np.zeros(len(q[k]), np.int64), 0
            # B: query k-grams x completions, so presence @ B = shared k-grams per (story, completion)
            cols = [q[k][g] for s in sets[k] for g in s]
            owner = [j for j, s in enumerate(sets[k]) for _ in s]
            B = sparse.csc_matrix((np.ones(len(cols), np.float32), (cols, owner)), shape=(len(q[k]), len(flat)))
            best = np.zeros(len(flat))
            jobs = [(corpus[i:i + chunk], k) for i in range(0, len(corpus), chunk)]
            for n, (c, t, pres) in enumerate(pool.imap(_scan, jobs)):
                counts += c; total += t
                if k in closest_ks:
                    shared = (pres @ B).toarray()
                    for j, (m, i, _) in enumerate(flat):  # the real ending's own story doesn't count
                        if m == "original" and n * chunk <= src_idx[i] < (n + 1) * chunk:
                            shared[src_idx[i] - n * chunk, j] = 0
                    best = np.maximum(best, shared.max(0))
            logf = {g: (np.log10(counts[c] / total) if counts[c] else -10.0) for g, c in q[k].items()}
            out["freq"][k] = {m: [logf[g] for j, (mm, _, _) in enumerate(flat) if mm == m for g in sets[k][j]]
                              for m in comps}
            if k in closest_ks:
                sizes = np.array([max(len(s), 1) for s in sets[k]])
                out["closest"][k] = {m: [float(best[j] / sizes[j]) for j, (mm, _, _) in enumerate(flat) if mm == m]
                                     for m in comps}
            print(f"k={k}: {len(q[k]):,} query k-grams, {total:,} corpus k-grams", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="dir with our trained models; TinyStories checkpoints always run")
    ap.add_argument("--models", help="comma list, default: every available model")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/overlap.json")
    ap.add_argument("--corpus_limit", type=int, help="first N training stories only, for a quick check")
    a = ap.parse_args()

    from datasets import load_dataset
    corpus = load_dataset("roneneldan/TinyStories", split="train")["text"][:a.corpus_limit]
    rng = np.random.default_rng(0)
    src = [int(i) for i in rng.choice(len(corpus), 4 * a.n, replace=False) if len(words(corpus[i])) >= 30][:a.n]
    prefixes, endings = zip(*(cut(corpus[i]) for i in src))

    gen_path = a.out.replace(".json", "_completions.json")
    gens = json.load(open(gen_path)) if os.path.exists(gen_path) else {}
    gens["original"] = list(endings)
    for name in (a.models.split(",") if a.models else available(a.ckpt)):
        if name not in gens:
            m = load(name, a.ckpt)
            gens[name] = [m.generate(p, max_new=200) for p in prefixes]
            print(name, "generated", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(gens, open(gen_path, "w"))

    comps = {m: [words(t) for t in ts] for m, ts in gens.items()}
    end2 = [grams(w, 2) for w in comps["original"]]
    res = {"stories": src, "fig14": {}, "fig15": {}}
    for m, ws in comps.items():
        g2 = [grams(w, 2) for w in ws]
        if m != "original":
            res["fig14"][m] = [prec(g, e) for g, e in zip(g2, end2)]
        res["fig15"][m] = [max(f1(g, h) for j, h in enumerate(g2) if j != i) for i, g in enumerate(g2)]
    stats = corpus_stats(corpus, comps, src)
    res["fig16"], res["fig17"] = stats["freq"], stats["closest"]
    json.dump(res, open(a.out, "w"))
    for m in comps:
        print(f"{m:14s} fig14 {np.mean(res['fig14'].get(m, [np.nan])):.3f}  fig15 {np.mean(res['fig15'][m]):.3f}  "
              f"fig17(k=2) {np.mean(res['fig17'][2][m]):.3f}")
