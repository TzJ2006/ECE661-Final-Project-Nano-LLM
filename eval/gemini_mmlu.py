"""MMLU for the Gemini API models on a fixed subset (20 questions per subject = 1,140), zero-shot,
answer by letter. Every response is cached in results/gemini_mmlu/<model>.jsonl and never re-sent.

python -m eval.gemini_mmlu --limit 20                 # trial: first 20 questions, check cost
python -m eval.gemini_mmlu                            # full subset, both models
"""
import argparse, json, os, pathlib, re
from concurrent.futures import ThreadPoolExecutor

MODELS = ["gemini-3.8-flash", "gemini-4-argon"]
BASE_URL = "https://34-127-166-57.sslip.io/api/v1"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def subset(per_subject=20, seed=0):
    """Same questions every run: the first `per_subject` of each subject after a seeded shuffle."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
    seen, out = {}, []
    for i, r in enumerate(ds):
        if seen.get(r["subject"], 0) < per_subject:
            seen[r["subject"]] = seen.get(r["subject"], 0) + 1
            out.append({"id": f"{r['subject']}-{i}", **r})
    return sorted(out, key=lambda r: r["id"])


def prompt(r):
    subj = r["subject"].replace("_", " ")
    choices = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
    return (f"The following is a multiple choice question about {subj}.\n\n{r['question']}\n{choices}\n"
            "Answer with the letter (A, B, C or D) only.")


def letter(text):
    m = re.search(r"\b([ABCD])\b", text or "")
    return m.group(1) if m else None


def run(model, rows, client, workers=8):
    path = ROOT / "results" / "gemini_mmlu" / f"{model}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]

    def ask(r):
        resp = client.chat.completions.create(model=model, temperature=0,
                                              messages=[{"role": "user", "content": prompt(r)}])
        text = resp.choices[0].message.content
        return {"id": r["id"], "subject": r["subject"], "gold": "ABCD"[r["answer"]], "pred": letter(text),
                "raw": text, "usage": resp.usage.model_dump() if resp.usage else None}

    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(ask, todo):
            f.write(json.dumps(rec) + "\n"); f.flush()
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = [x for x in recs if x["id"] in {r["id"] for r in rows}]
    acc = sum(x["pred"] == x["gold"] for x in recs) / len(recs)
    toks = sum((x["usage"] or {}).get("total_tokens", 0) for x in recs)
    print(f"{model}: {len(todo)} new calls, {len(recs)} scored, acc {acc:.3f}, {toks:,} tokens total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, help="only the first N subset questions (trial run)")
    a = ap.parse_args()
    from openai import OpenAI
    key = os.environ.get("ECE661_API_KEY") or json.loads((ROOT / "ece661-key.json").read_text())["key"]
    client = OpenAI(base_url=BASE_URL, api_key=key)
    rows = subset()[:a.limit]
    for m in a.models.split(","):
        run(m, rows, client)
