"""Label the Figs 9-11 completions with Gemini (success / partial / failure), blind to the model.
The paper labeled by hand; we check Gemini against a human spot-check (results/figs9_11_spotcheck.csv).

python -m eval.label_figs9_11 --limit 10      # trial
python -m eval.label_figs9_11                 # all rows, cached in results/figs9_11_gemini.jsonl
"""
import argparse, csv, hashlib, json, os, pathlib, random, re
from concurrent.futures import ThreadPoolExecutor

from eval.gemini_mmlu import BASE_URL

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "figs9_11_all.csv"
CACHE = ROOT / "results" / "figs9_11_gemini.jsonl"
JUDGE = "gemini-4-argon"
TESTS = {"factual": "Does the continuation state the correct fact or the commonsense answer the prompt is asking for?",
         "reasoning": "Does the continuation follow logically from the situation in the prompt?",
         "context-tracking": "Does the continuation stay consistent with who, what and how many the prompt set up?"}


def judge_prompt(row):
    kind = row["figure"].split(" ")[0]
    return ("You are grading a tiny language model on a children's story prompt, as in the TinyStories paper "
            f"(Eldan & Li, 2023). Test: {TESTS[kind]}\n"
            "Judge only the first sentence or clause of the continuation, where the answer is; ignore later text "
            "and grammar slips that do not change the answer.\n\n"
            f"PROMPT: {row['prompt']}\nCONTINUATION: {row['completion'].strip()}\n\n"
            "Reply with exactly one line: <label> | <one-sentence reason>, where <label> is success (correct), "
            "partial (plausible but not the expected answer, or only partly right) or failure (wrong or nonsensical).")


def key(row):
    return hashlib.sha1((row["prompt"] + "\x00" + row["completion"]).encode()).hexdigest()[:16]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from openai import OpenAI
    k = os.environ.get("ECE661_API_KEY") or json.loads((ROOT / "ece661-key.json").read_text())["key"]
    client = OpenAI(base_url=BASE_URL, api_key=k)

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    cache = {json.loads(l)["key"]: json.loads(l) for l in open(CACHE, encoding="utf-8")} if CACHE.exists() else {}
    todo = [r for r in rows[:a.limit] if key(r) not in cache]

    def ask(r):
        text = client.chat.completions.create(model=JUDGE, temperature=0, messages=[
            {"role": "user", "content": judge_prompt(r)}]).choices[0].message.content.strip()
        m = re.match(r"\W*(success|partial|failure)\W*\|?\s*(.*)", text, re.I | re.S)
        return {"key": key(r), "label": m.group(1).lower() if m else None, "reason": (m.group(2) if m else text).strip()}

    with open(CACHE, "a", encoding="utf-8") as f, ThreadPoolExecutor(8) as ex:
        for rec in ex.map(ask, todo):
            cache[rec["key"]] = rec
            f.write(json.dumps(rec) + "\n")

    for r in rows:
        c = cache.get(key(r), {})
        r["label_gemini"], r["reason_gemini"] = c.get("label", ""), c.get("reason", "")
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader(); w.writerows(rows)

    done = [r for r in rows if r["label_gemini"]]
    print(f"{len(todo)} new calls, {len(done)}/{len(rows)} labeled, "
          f"{sum(not r['label_gemini'] for r in rows[:a.limit])} unparsed in range")
    spot = ROOT / "results" / "figs9_11_spotcheck.csv"
    if spot.exists():  # agreement between your hand labels and Gemini's, once you have filled them in
        human = [r for r in csv.DictReader(open(spot, encoding="utf-8")) if r["label"].strip()]
        if human:
            g = {(r["figure"], r["prompt_id"], r["model"]): r["label_gemini"] for r in rows}
            agree = sum(r["label"].strip().lower() == g[(r["figure"], r["prompt_id"], r["model"])] for r in human)
            print(f"spot-check agreement: {agree}/{len(human)} = {agree / len(human):.0%}")
    elif len(done) == len(rows):
        pick = random.Random(0).sample(rows, 30)  # human labels go in `label`; Gemini's stay hidden here
        with open(spot, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["figure", "prompt_id", "model", "prompt", "completion", "label"])
            w.writeheader(); w.writerows({c: r[c] for c in w.fieldnames} for r in pick)
        print("wrote", spot)
