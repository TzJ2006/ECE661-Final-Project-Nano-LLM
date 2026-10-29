"""Demo: stories from every model, side by side, for the same prompt.

Serves cached stories (results/demo_cache.json) so it runs offline at the poster session.
python app.py                                   # serve the demo
python app.py --build --ckpt ~/project/ckpt     # (TPU VM) generate stories for our + TinyStories models
python app.py --build --gemini                  # (laptop) add Gemini stories through the course API
"""
import argparse, json, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / "results" / "demo_cache.json"
PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She loved to",
    "Tom found a shiny key in the garden. He wondered",
    "The cat and the dog were best friends. One day, they",
    "It was raining outside, so Ben and his mom decided to",
    "Sara wanted to bake a cake for her grandma, but",
    "A small bird could not fly. Every day it watched the other birds and",
    "Max had a red ball. He threw it so high that",
    "Anna was scared of the dark. One night, she heard a noise and",
    "The little boat sailed on the big blue sea until",
    "Lucy and her brother found a box in the attic. Inside the box was",
    "Once upon a time, a friendly dragon lived on a hill. The dragon",
    "Tim did not want to share his toys with his sister. His mom said,",
]
ORDER = ["1M", "3M", "8M", "28M", "33M", "1Layer-21M", "2Layers-33M",
         "baseline", "looped", "mdlm", "looped_mdlm", "gemini-3.8-flash", "gemini-4-argon"]
LABEL = {**{m: f"TinyStories-{m}" for m in ORDER[:7]}, "baseline": "Ours: transformer (33M)",
         "looped": "Ours: looped transformer (33M)", "mdlm": "Ours: diffusion (33M)",
         "looped_mdlm": "Ours: looped diffusion (33M)", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "gemini-4-argon": "Gemini 4 Argon"}


def build_local(ckpt):
    from eval.models import available, load
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for name in available(ckpt):
        m = load(name, ckpt)
        for p in PROMPTS:
            cache.setdefault(p, {})[name] = m.generate(p, max_new=200)
        print(name, "done", flush=True)
    CACHE.write_text(json.dumps(cache, indent=1))


def build_gemini():
    from openai import OpenAI
    from eval.gemini_mmlu import BASE_URL
    key = os.environ.get("ECE661_API_KEY") or json.loads((ROOT / "ece661-key.json").read_text())["key"]
    client = OpenAI(base_url=BASE_URL, api_key=key)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for model in ["gemini-3.8-flash", "gemini-4-argon"]:
        for p in PROMPTS:
            if model in cache.get(p, {}):
                continue
            r = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "user", "content":
                "Continue this short story for a young child in simple words, in about 150 words. "
                f"Reply with the continuation only.\n\n{p}"}])
            cache.setdefault(p, {})[model] = " " + r.choices[0].message.content.strip()
        print(model, "done", flush=True)
    CACHE.write_text(json.dumps(cache, indent=1))


def serve():
    import gradio as gr
    cache = json.loads(CACHE.read_text())
    prompts = [p for p in PROMPTS if p in cache]

    def show(prompt):
        stories = cache[prompt]
        return "\n\n".join(f"### {LABEL[m]}\n{prompt}**{stories[m]}**" for m in ORDER if m in stories)

    with gr.Blocks(title="Tiny language models") as demo:
        gr.Markdown("# How well do tiny models write?\nThe same story opening, continued by models from "
                    "1M parameters to Gemini. Bold text is written by the model.")
        pick = gr.Dropdown(prompts, value=prompts[0], label="Story opening")
        out = gr.Markdown(show(prompts[0]))
        pick.change(show, pick, out)
    demo.launch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--gemini", action="store_true")
    a = ap.parse_args()
    if not a.build:
        serve()
    elif a.gemini:
        build_gemini()
    else:
        build_local(a.ckpt)
