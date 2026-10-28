# ECE661 Project Proposal: Tiny Language Models (Topic 9)

**Team:** Thomas Tang, Michael Wang · ECE661, Fall 2026 · Topic 9 (Tiny Language Models)

## 1. What are we trying to do?

We will run a controlled comparison of four types of language model at the tiny scale where coherent writing first emerges, and place them on a full size scale up to frontier API models such as Gemini. Every model gets the same tests: MMLU, LLM-judged story writing, and a set of standard benchmarks. Concretely, we will:

1. Reproduce the TinyStories paper's qualitative results (Figs 9–11) and memorization analysis (Sec 4.4.1, Figs 14–17) using the released checkpoints. This validates our evaluation pipeline against published results.
2. Train four model types at **5M, 15M and 33M parameters**, the range where TinyStories showed coherent writing emerges: a standard **transformer** (baseline), a **looped transformer** (one shared block applied several times), a **masked diffusion LM** (writes by iteratively unmasking, LLaDA/MDLM style), and **Mamba** (state space). We also train one hybrid, a looped diffusion LM.
3. Place our models on one size scale with open pretrained models (TinyStories 1M–33M, Pythia 70M–6.9B, SmolLM2 135M–1.7B) and API models (the Gemini family through the course API, from 2.5 Flash-Lite to 3.1 Pro and Gemini 4 Argon).
4. **Headline comparisons:** MMLU accuracy and LLM-judged writing quality across the whole scale, with Gemini as the frontier reference. Supporting benchmarks: BLiMP, HellaSwag, PIQA, ARC-Easy, LAMBADA.
5. Build a web demo that shows stories from every model side by side for the same prompt.

## 2. How is it done today, and what are the limits?

Useful language models today usually have billions of parameters and are trained on web-scale text. TinyStories (Eldan & Li, 2023) showed that transformers with 1–33M parameters, trained on a synthetic corpus written with a child's vocabulary, can produce fluent and consistent short stories. Its authors' follow-ups at Microsoft (phi-1, phi-1.5, Phi-3) carried the synthetic-data idea to 1–4B models. Meta's MobileLLM (2024) showed that architecture choices such as depth and weight sharing matter below 1B parameters.

What that work leaves open:

- **Only one model type is tested at tiny scale.** Every TinyStories model is a left-to-right transformer; the paper varies only its depth and width.
- **New model types have been shown only at large scale.** Masked diffusion LMs (MDLM, LLaDA), looped transformers and state-space models (Mamba) each challenge the standard transformer, but mostly at hundreds of millions of parameters or more. To our knowledge, no prior work compares them head-to-head at 1–33M parameters.
- **No single scale links tiny models to frontier ones.** TinyStories reports judged writing scores, while MMLU and similar benchmarks are reported for large models. To our knowledge, no study places both measures on one scale running from 1M parameters to a frontier API model.

## 3. What is new in our approach, and why will it work?

Our approach is new in putting four model types and their hybrids through the same fair comparison: same data, tokenizer, parameter budgets (5M, 15M, 33M), token budget and evaluation. Each type changes one independent part of the transformer:

- how text is generated: left to right vs. iterative unmasking (diffusion)
- how depth is obtained: separate layers vs. one shared block looped
- how tokens mix: attention vs. a state-space layer

Because these parts are independent, they can be combined into hybrids, and each comparison changes just one of them.

**Pipeline.** All training runs on the course's Google Cloud TPU (one v6e chip per session). We will write one shared JAX/Flax codebase, ported from nanoGPT's design, in which the model type is a config switch; JAX is native to TPUs. Every run shares the same tokenizer, token budget, optimizer, learning-rate schedule, context length and seeds.

**Models we train** (each at 5M, 15M and 33M parameters):

| Model | Change from baseline | How it writes | Question it answers |
| --- | --- | --- | --- |
| Transformer (baseline) | None: LLaMA-style block (RoPE, RMSNorm, SwiGLU) | Left to right, one token per step | Reference point |
| Looped transformer | A few shared blocks applied several times | Left to right | Can reused depth stand in for parameters? Does looping more at test time help? |
| Masked diffusion LM | No causal mask; trained to fill in randomly masked tokens | Starts fully masked, unmasks over N steps | Can a tiny diffusion model write coherent stories, and how many steps does it need? |
| Mamba (state space) | Attention replaced by selective state-space layers | Left to right, constant memory per token | Is attention needed at this scale? |
| Hybrid: looped diffusion | Diffusion objective on the looped backbone | Unmasking | Does extra depth help diffusion more than it helps left-to-right models? |

**Fairness.** A looped model has few parameters but does many layers of compute per token. We therefore also train a compute-matched control (8 separate layers) and report each model's parameters and FLOPs per token side by side.

**Comparison scale.** We put our models on one axis of model size next to models we do not train:

| Tier | Models | Size | Role |
| --- | --- | --- | --- |
| Ours | 4 types + 1 hybrid | 5M / 15M / 33M | The controlled study |
| TinyStories checkpoints | GPT-Neo, trained on TinyStories | 1M–33M | Same data; the paper's own reference |
| Open pretrained | Pythia suite; SmolLM2 | 70M–6.9B; 135M–1.7B | General-text models of known size |
| API | Gemini 2.5 Flash-Lite, 2.5 Flash, 2.5 Pro, 3.x Flash, 3.1 Pro, Gemini 4 Argon | Not disclosed | Frontier ceiling; also serve as judges |

**Benchmarks:**

| Test | Measures | Run on | What it establishes |
| --- | --- | --- | --- |
| **MMLU (headline)** | Knowledge across 57 subjects, 4 choices | All tiers (API: sampled subset) | Where along the size scale broad knowledge emerges above the 25% chance line |
| **Story writing, LLM-judged (headline)** | Grammar, creativity, consistency, plot, scored 1–10 | All tiers, same prompts | Our core arena: how much of frontier writing quality tiny models deliver at a small fraction of the size |
| BLiMP | Grammar, via minimal sentence pairs | Ours + open models | Grammatical competence at every size; separates model types where knowledge benchmarks cannot |
| HellaSwag, PIQA, ARC-Easy, LAMBADA | Common sense and word prediction | All tiers | How common sense grows along the size scale |
| Sec 4.4.1 overlap | Memorization of training stories | Ours + TinyStories checkpoints | Rules out memorization as the source of fluent stories |
| Validation loss, generation speed | Fit to data, cost | Ours | Training efficiency and generation cost of each model type |

- **Tooling:** our models are scored by our own JAX evaluation code, which uses the same prompt formats as EleutherAI's `lm-evaluation-harness`. Open models run through the harness itself (PyTorch/XLA on the TPU VM). For diffusion models, multiple-choice answers are scored with the likelihood bound, as LLaDA does.
- **API models:** all API models are Gemini, called through the course's OpenAI-compatible endpoint. Gemini models answer by letter, the standard protocol for API-only models. Having several Gemini generations (Flash-Lite, Flash, Pro, Argon) also gives a small size scale within the API tier.
- **Judging:** several Gemini generations judge each story; no model judges its own outputs, judge agreement is reported, and human ratings on about 50 stories calibrate the judges.

**Demo.** A Gradio web app hosted on Hugging Face Spaces. A user enters a story prompt and sees stories from models across the scale side by side, along with the judges' scores. For the poster session we will also cache outputs, so the demo works without network access.

**Extensions.** A second hybrid (attention + Mamba layers); quantizing the best model to int8/int4 (ties to ECE661 compression); the topic's top-K token-filtering experiment.

**Why it should work.** TinyStories already shows that 1–33M transformers learn this corpus, which gives us a working baseline. Each other type is a small change to the same code: looping changes the forward pass, diffusion changes the loss and the sampler, and Mamba swaps the token-mixing layer. Mamba's selective scan is written with `jax.lax.associative_scan`, which runs natively on TPU. A 33M model on the full corpus needs about 10^17 FLOPs, well under an hour on one v6e chip. For the larger models we only run evaluation, using existing checkpoints and the Gemini API, so they add no training cost.

## 4. Who cares? If we succeed, what difference will it make?

People building on-device language features need to know what a tiny local model delivers compared with calling an API such as Gemini, and which model type gets the most out of a fixed memory budget. Our size scale answers both questions on the same tests: how much of frontier writing quality tiny models deliver, where general knowledge (MMLU) starts to appear, and whether diffusion, looping or state-space layers beat a standard transformer at small sizes. The demo lets anyone at the poster session compare the models side by side.

## 5. What are the risks?

| Risk | Mitigation |
| --- | --- |
| Schedule across about 16 training runs | Core four types train first; the hybrid and extensions build on them |
| TPU session limits: 1 chip, at most 8 hours, deleted after 30 idle minutes, disk wiped at the end | Each run fits in one session; checkpoints saved to `/mnt/group` regularly; save, stop and resume tested on a small job before any long run |
| Gemini judges favor Gemini-written stories | Never let a model judge its own outputs; several Gemini generations as judges; human ratings on about 50 stories to check them |
| Shared group Gemini budget | Sampled MMLU subset for API models, cached responses, Flash-Lite for development, usage checked on the course page |
| API scores not comparable with local scores (no token probabilities) | Score Gemini by answer letter on a fixed subset; report the method difference |
| Running PyTorch checkpoints on TPU | Run them through PyTorch/XLA; for the largest Pythia models, use published scores if needed |
| Mamba's fast kernels need NVIDIA GPUs | Selective scan written in JAX (`jax.lax.associative_scan`) |
| Diffusion LMs need more sampling steps at small sizes | A step-count sweep maps the quality–speed trade-off |

## 6. How much will it cost?

We estimate 30–60 TPU-hours, or about $40–80 of our $150 TPU budget (one v6e chip at $1.35/hour). Gemini calls draw on a separate $100 Gemini API budget. Training is the small part. A 33M model on the full corpus of about 0.5B tokens needs about 10^17 FLOPs, well under an hour on one v6e chip. About 16 training runs (4 types × 3 sizes, the hybrid, the compute-matched control and repeat seeds) fit in about 10–20 TPU-hours. Most of the rest goes to evaluating open models on MMLU and the other benchmarks.

- **Hardware:** the course TPU, one v6e chip per session (32 GB memory, sessions of at most 8 hours). Pythia-6.9B, our largest open model, fits on one chip.
- **Data and checkpoints:** TinyStories dataset and checkpoints, Pythia and SmolLM2 checkpoints, all free on Hugging Face.
- **Software:** JAX/Flax (our models), PyTorch/XLA with Hugging Face `transformers` and `lm-evaluation-harness` (open models), the `openai` client (course Gemini endpoint), `rouge-score`, Gradio.
- **API:** the Gemini models on the course endpoint, used for benchmark answers, story generation and judging. $100 budget, tracked on the course usage page.

## 7. How long will it take?

Six weeks, from Oct 26 to the in-person poster session on Dec 9, 2026. Training, replication and benchmarking of the open and API models run in parallel, so the midterm check-in can already show results.

| Work stream | Dates (2026) |
| --- | --- |
| Proposal + setup | Oct 26 – Nov 1 |
| Replicate Figs 9–17 | Oct 29 – Nov 13 |
| Pipeline + baseline | Nov 2 – Nov 13 |
| Benchmarks + LLM judges | Nov 4 – Nov 27 |
| Looped + Mamba, 3 sizes | Nov 9 – Nov 24 |
| Diffusion LM, 3 sizes | Nov 9 – Nov 24 |
| Looped diffusion hybrid | Nov 18 – Nov 27 |
| Gradio demo | Nov 23 – Dec 4 |
| Report + poster | Nov 30 – Dec 8 |
| Poster session | Dec 9 |

Open and API models need no training, so their benchmarks start in week 2. Our model types train at three sizes from Nov 9. The hybrid builds on the trained looped and diffusion models in the final training week.

## 8. Expected outcomes

- A controlled comparison of four model types at 5M, 15M and 33M parameters under one recipe, plus the looped diffusion hybrid and a compute-matched control.
- **Headline figure 1:** MMLU accuracy against model size, from our 5M models to Gemini, locating where broad knowledge emerges.
- **Headline figure 2:** judged writing quality against model size on the same scale, showing how much frontier writing quality tiny models deliver.
- The model type that writes best per parameter at tiny scale, with BLiMP, HellaSwag, PIQA, ARC-Easy, LAMBADA, validation loss and generation speed supporting each comparison.
- A live Gradio demo, the final report and poster, and a public code repository.

## Team roles

| Member | Owns | Shared |
| --- | --- | --- |
| Thomas Tang | Replication of Figs 9–11 and 14–17; benchmark pipeline (`lm-evaluation-harness`, MMLU, open and API models); LLM-judge pipeline; Gradio demo | Report, poster, Q&A preparation |
| Michael Wang | Shared codebase and baseline; training all model types at three sizes (transformer, looped, masked diffusion, Mamba, hybrid) | Report, poster, Q&A preparation |

Thomas and Michael review each other's code and results, so both can answer questions on the full project.

## References

1. Eldan, R. and Li, Y. "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
2. TinyStories checkpoints and dataset: [huggingface.co/roneneldan](https://huggingface.co/roneneldan)
3. Gunasekar, S. et al. "Textbooks Are All You Need" (phi-1), Microsoft. [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)
4. Liu, Z. et al. "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases", Meta. [arXiv:2402.14905](https://arxiv.org/abs/2402.14905)
5. Nie, S. et al. "Large Language Diffusion Models" (LLaDA). [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
6. Sahoo, S. et al. "Simple and Effective Masked Diffusion Language Models" (MDLM). [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)
7. Dehghani, M. et al. "Universal Transformers." [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
8. Saunshi, N. et al. "Reasoning with Latent Thoughts: On the Power of Looped Transformers." [arXiv:2502.17416](https://arxiv.org/abs/2502.17416)
9. Gu, A. and Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
10. Biderman, S. et al. "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling." [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
11. Allal, L. B. et al. "SmolLM2: When Smol Goes Big." [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)
12. Hendrycks, D. et al. "Measuring Massive Multitask Language Understanding" (MMLU). [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
13. Warstadt, A. et al. "BLiMP: The Benchmark of Linguistic Minimal Pairs." [arXiv:1912.00582](https://arxiv.org/abs/1912.00582)
14. EleutherAI lm-evaluation-harness: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
15. Google Gemini API: [ai.google.dev](https://ai.google.dev)
16. Karpathy, A. nanoGPT: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
