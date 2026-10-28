"""Ask Gemini 4 Argon about paper files.

python argon.py "Review this as a CVPR reviewer" main.tex sec/method.tex
python argon.py -m gemini-2.5-flash "Fix grammar only" sec/intro.tex > intro_fixed.txt
"""
import argparse, json, os, pathlib
from openai import OpenAI

SYSTEM = ("You help a student write a CVPR paper. Be precise and concise. "
          "Never invent results, numbers, citations, or references; if something is missing, say so.")

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("files", nargs="*")
ap.add_argument("-m", "--model", default="gemini-4-argon")
a = ap.parse_args()

key = os.environ.get("ECE661_API_KEY") or json.loads(pathlib.Path(__file__).with_name("ece661-key.json").read_text())["key"]
client = OpenAI(base_url="https://34-127-166-57.sslip.io/api/v1", api_key=key)

docs = "\n\n".join(f"=== {f} ===\n{pathlib.Path(f).read_text(encoding='utf-8')}" for f in a.files)
r = client.chat.completions.create(model=a.model, messages=[
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"{a.prompt}\n\n{docs}" if docs else a.prompt},
])
print(r.choices[0].message.content)
