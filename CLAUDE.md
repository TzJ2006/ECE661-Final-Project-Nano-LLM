# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ECE661 (Fall 2026) course project, Topic 9: tiny language models. Four ~33M-parameter models are trained on TinyStories in JAX/Flax and compared against the released TinyStories checkpoints (1M–33M, PyTorch/HF) and Gemini API models. `proposal.md` is the plan; `results/summary.md` is the current results table. The proposal also lists 5M/15M sizes and Mamba, which are not implemented.

## Where things run

- **Training and evaluation of local models run on the course TPU VM** (one v6e chip, sessions of at most 8 h, disk deleted when the session ends). Checkpoints and tokenized data live outside the repo (`~/project/ckpt`, `~/project/data` on the VM; `data/`, `checkpoints/`, `*.bin` are gitignored). `ECE661-API使用教程.md` documents the `ece661` CLI, TPU sessions and the API.
- **The laptop (Windows) only does**: Gemini API calls, `eval.figures`, the demo, and `--tiny` debug runs. Everything under `results/` is committed so figures and the demo work without a TPU.
- **Gemini** goes through the course's OpenAI-compatible endpoint (`BASE_URL` in `eval/gemini_mmlu.py`). The key comes from `ECE661_API_KEY` or the gitignored `ece661-key.json`. The budget is shared by the whole group: every Gemini script caches responses in `results/` and never re-sends them, and has a `--limit` for a trial run first. Keep that behaviour.

## Commands

All modules are run from the repo root with `python -m`. There is no test suite or linter; the checks are the debug runs below.

```bash
pip install -r requirements.txt          # on the TPU VM first: pip install -U "jax[tpu]"

python -m tinylm.configs                 # print params and FLOPs/token of the four configs
python -m tinylm.data --out data --limit 2000                               # small local token files
python -m tinylm.train mdlm --tiny --overfit --data data --out ckpt/tiny    # smoke test: loss -> ~0 in seconds
python -m tinylm.ckpt_test --dir ~/project/ckpt_test                        # run in two TPU sessions: save, then restore

# full pipeline (TPU VM)
python -m tinylm.data --out ~/project/data
python -m tinylm.train baseline --data ~/project/data --out ~/project/ckpt/baseline   # also: looped, mdlm, looped_mdlm
python -m tinylm.sample ~/project/ckpt/mdlm "Once upon a time" --steps 64
bash eval/run_harness.sh ts                    # TinyStories checkpoints through lm-eval (CPU)
bash eval/run_harness.sh ours ~/project/ckpt   # our four models; TASKS=mmlu,blimp to pick tasks
python -m eval.harness_lm ~/project/ckpt/baseline --tasks mmlu --limit 50 --out results/harness/x   # one model, quick
python -m eval.figs9_11 --ckpt ~/project/ckpt
python -m eval.overlap --ckpt ~/project/ckpt --out results/overlap.json
python app.py --build --ckpt ~/project/ckpt    # demo stories from local models

# laptop
python -m eval.gemini_mmlu --limit 20    # trial, then without --limit
python -m eval.label_figs9_11 --limit 10 # Gemini judge for Figs 9-11, then without --limit
python app.py --build --gemini
python -m eval.figures                   # all figures + results/summary.md from results/ only
python app.py                            # serve the demo from results/demo_cache.json
```

## Architecture

**One model, two switches** (`tinylm/configs.py`, `tinylm/model.py`). Each config is `BASE` with one or two fields changed: `loop` (how many times the 4-block stack is applied; weights shared across passes, so parameters stay equal and only FLOPs grow) and `objective` (`"ar"` = causal mask, `"mdlm"` = bidirectional masked diffusion). The training recipe (batch, LR schedule, 500M-token budget, seed) is identical across models by design — the project is a controlled comparison, so don't tune one model separately.

**Token ids.** GPT-2 BPE via tiktoken, the same tokenizer as the TinyStories checkpoints. `MASK = 50257` is one id past the vocabulary; all four models have a 50258-row embedding so sizes match, and every sampling/scoring path slices logits with `[..., :MASK]` so MASK is never predicted.

**Training** (`tinylm/train.py`). Data is a flat uint16 memmap of tokens with EOS after each story; batches are random `(B, T+1)` windows. `train.py` resumes automatically from `OUT/ckpt.msgpack` (atomic save every `--ckpt_minutes`, because TPU sessions get killed), and writes `OUT/config.json`, `OUT/log.jsonl`, and at the end `OUT/params.msgpack`. A model counts as "trained" for eval/demo purposes iff `params.msgpack` exists (`eval.models.available`). Logs are copied by hand into `results/train_logs/`.

**Inference** (`tinylm/sample.py::Model`) is the single entry point for everything downstream. It loads from a training output dir and provides:
- `generate()` — AR: token by token, no KV cache. MDLM: start from MASKs, commit the most confident predictions each step (LLaDA low-confidence remasking). `loop=` overrides the loop count at test time.
- `loglik_batch()` — AR: exact. MDLM: Monte-Carlo ELBO (LLaDA eq. 6), exact for 1-token continuations. Diffusion inputs are always padded to `block_size` with MASK (the model only saw full windows; also keeps one jit shape).

**Evaluation** (`eval/`).
- `eval/models.py` puts HF TinyStories checkpoints and our JAX models behind the same `generate()`; `TS` and `OURS` are the canonical model-name lists used by figures and the demo.
- `eval/harness_lm.py` wraps `Model` as an lm-evaluation-harness `LM` so our models see exactly the prompts the HF checkpoints see; its tokenization of context/continuation deliberately mirrors the harness's `HFLM`. Only `loglikelihood` is implemented.
- Harness outputs have two layouts under `results/harness/`: lm-eval CLI layout for `ts-*`, and `<model>/<task>/results.json` for ours. `eval/figures.py::harness()` reads both.
- `figs9_11.py` → completions CSV → `label_figs9_11.py` (Gemini 4 Argon judge, blind to model, checked against `results/figs9_11_spotcheck.csv`). `overlap.py` reproduces the TinyStories memorization analysis (Figs 14–17).

**Conventions.** Scripts are flat and argparse-driven with the usage example in the module docstring; keep that docstring current when flags change. Deliberate simplifications are marked with a `# ponytail:` comment naming the ceiling and the upgrade path.

`argon.py` is a standalone helper for asking Gemini about paper files; it is unrelated to the pipeline.
