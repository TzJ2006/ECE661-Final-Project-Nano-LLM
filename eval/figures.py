"""Every report/poster figure and the results table, from results/ only (no TPU needed).

python -m eval.figures            # -> results/figures/*.png, results/summary.md
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.models import TS, OURS

R = "results"
FIG = os.path.join(R, "figures")
TIER = {"TinyStories": "#2a78d6", "Ours (33M)": "#eb6834", "Gemini": "#1baf7a"}  # validated, fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]                              # our four models
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
TASKS = [("mmlu", "MMLU"), ("blimp", "BLiMP"), ("hellaswag", "HellaSwag"), ("piqa", "PIQA"),
         ("arc_easy", "ARC-Easy"), ("lambada_openai", "LAMBADA")]
LABEL = {**{m: f"TS-{m}" for m in TS}, "baseline": "Transformer", "looped": "Looped",
         "mdlm": "Diffusion", "looped_mdlm": "Looped diffusion",
         "gemini-3.8-flash": "Gemini 3.8 Flash", "gemini-4-argon": "Gemini 4 Argon"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})


def harness():
    """{model: {task: acc}} from lm-eval outputs (CLI layout for TS-*, ours write results.json)."""
    out = {}
    for f in glob.glob(f"{R}/harness/*/**/results*.json", recursive=True):
        name = f.replace("\\", "/").split("/")[2].removeprefix("ts-")
        res = json.load(open(f))["results"]
        out.setdefault(name, {}).update({t: res[t].get("acc,none") for t, _ in TASKS if t in res})
    return out


def gemini():
    out = {}
    for f in glob.glob(f"{R}/gemini_mmlu/*.jsonl"):
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        out[os.path.basename(f)[:-6]] = sum(r["pred"] == r["gold"] for r in recs) / len(recs)
    return out


def headline(h, g):
    rows = [(LABEL[m], h[m]["mmlu"], "TinyStories") for m in TS if "mmlu" in h.get(m, {})]
    rows += [(LABEL[m], h[m]["mmlu"], "Ours (33M)") for m in OURS if "mmlu" in h.get(m, {})]
    rows += [(LABEL.get(m, m), a, "Gemini") for m, a in g.items()]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(rows))
    ax.bar(x, [100 * r[1] for r in rows], width=0.7, color=[TIER[r[2]] for r in rows], edgecolor="white", linewidth=2)
    for i, r in enumerate(rows):
        ax.text(i, 100 * r[1] + 1.5, f"{100 * r[1]:.0f}", ha="center", fontsize=8, color=INK)
    ax.axhline(25, ls="--", lw=1, color=MUTED)
    ax.text(-0.45, 31, "chance (25%)", ha="left", fontsize=8, color=MUTED)
    ax.set_xticks(x, [r[0] for r in rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MMLU accuracy (%)"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER.values()], labels=list(TIER),
              frameon=False, loc="upper left")
    ax.set_title("MMLU from 1M parameters to frontier models (zero-shot; Gemini on a 1,140-question subset)",
                 fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/mmlu_scale.png", dpi=200); plt.close(fig)


def table(h, g):
    lines = ["| Model | " + " | ".join(n for _, n in TASKS) + " |", "|---" * (len(TASKS) + 1) + "|"]
    for m in [m for m in TS + OURS if m in h]:
        lines.append(f"| {LABEL[m]} | " + " | ".join(f"{100 * h[m][t]:.1f}" if h[m].get(t) is not None else "–"
                                                     for t, _ in TASKS) + " |")
    for m, a in g.items():
        lines.append(f"| {LABEL.get(m, m)} | {100 * a:.1f} (subset) |" + " – |" * (len(TASKS) - 1))
    note = ("Zero-shot accuracy (%). Diffusion models are scored with the LLaDA likelihood bound: exact for "
            "1-token answers (MMLU), 16 Monte-Carlo samples otherwise. Gemini answers by letter.\n\n")
    open(f"{R}/summary.md", "w").write(note + "\n".join(lines) + "\n")


def loss_curves():
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in zip(OURS, SERIES):
        f = f"{R}/train_logs/{m}.jsonl"
        if os.path.exists(f):
            v = [json.loads(l) for l in open(f) if '"val"' in l][1:]  # skip the untrained step-0 point
            ax.plot([r["tokens"] / 1e6 for r in v], [r["val"] for r in v], lw=2, color=c,
                    label=f"{LABEL[m]} ({v[-1]['val']:.2f})")
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("Validation loss (diffusion: upper bound on NLL)", fontsize=10, loc="left")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/loss.png", dpi=200); plt.close(fig)


def overlap():
    f = f"{R}/overlap.json"
    if not os.path.exists(f):
        return
    o = json.load(open(f))
    specs = [("fig14", "Fig 14: ROUGE-2 precision vs the real ending", o["fig14"], (0, 1)),
             ("fig15", "Fig 15: max ROUGE-2 F vs the model's other completions", o["fig15"], (0, 1)),
             ("fig17_k2", "Fig 17: ROUGE-2 precision vs the closest training story", o["fig17"]["2"], (0, 1)),
             ("fig17_k3", "Fig 17: ROUGE-3 precision vs the closest training story", o["fig17"]["3"], (0, 1))]
    specs += [(f"fig16_k{k}", f"Fig 16: log10 corpus frequency of each {k}-gram (-10 = never seen)",
               o["fig16"][str(k)], (-10.5, 0)) for k in (2, 3, 4, 5)]
    for name, title, data, rng in specs:
        models = [m for m in TS + OURS + ["original"] if m in data]
        fig, axs = plt.subplots(1, len(models), figsize=(1.6 * len(models), 2.2), sharey=True)
        for ax, m in zip(np.atleast_1d(axs), models):
            ax.hist(data[m], bins=20, range=rng, color=SERIES[1] if m in OURS else SERIES[0], edgecolor="white")
            ax.set_title(LABEL.get(m, m), fontsize=8); ax.tick_params(labelsize=7)
        fig.suptitle(title, fontsize=10, x=0.01, ha="left"); fig.tight_layout()
        fig.savefig(f"{FIG}/{name}.png", dpi=200); plt.close(fig)


def figs9_11():
    """Paper-style grid: prompt x model, cell = label (letter + status color), plus a success count row."""
    import csv
    f = f"{R}/figs9_11_all.csv"
    if not os.path.exists(f):
        return
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    if not any(r.get("label_gemini") for r in rows):
        return
    col = {"success": "#0ca30c", "partial": "#fab219", "failure": "#d03b3b"}  # status palette, never alone
    models = [m for m in TS + OURS if any(r["model"] == m for r in rows)]
    prompts = list(dict.fromkeys((r["figure"], r["prompt_id"]) for r in rows))
    lab = {(r["figure"], r["prompt_id"], r["model"]): r["label_gemini"] for r in rows}
    fig, ax = plt.subplots(figsize=(1 + 0.75 * len(models), 0.3 * (len(prompts) + 2)))
    for y, p in enumerate(prompts):
        for x, m in enumerate(models):
            l = lab.get((*p, m), "")
            ax.add_patch(plt.Rectangle((x, y), 0.94, 0.9, color=col.get(l, GRID)))
            ax.text(x + 0.47, y + 0.45, l[:1].upper(), ha="center", va="center", fontsize=7,
                    color=INK if l == "partial" else "white")
    y = len(prompts) + 0.3
    for x, m in enumerate(models):
        ax.text(x + 0.47, y + 0.45, sum(lab.get((*p, m)) == "success" for p in prompts),
                ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, len(models)); ax.set_ylim(len(prompts) + 1.4, 0); ax.grid(False)
    ax.set_xticks([x + 0.47 for x in range(len(models))], [LABEL[m] for m in models], rotation=40, ha="right", fontsize=8)
    ax.set_yticks([y + 0.45 for y in range(len(prompts))] + [len(prompts) + 0.75],
                  [f"{fig_.split(' ')[0][:4]}. {int(i) + 1}" for fig_, i in prompts] + ["# success"], fontsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Figs 9-11: S = success, P = partial, F = failure (labels by Gemini 4 Argon, blind to model)",
                 fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/figs9_11.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    h, g = harness(), gemini()
    headline(h, g); table(h, g); loss_curves(); overlap(); figs9_11()
    print(open(f"{R}/summary.md").read())
