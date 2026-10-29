Zero-shot accuracy (%). Diffusion models are scored with the LLaDA likelihood bound: exact for 1-token answers (MMLU), 16 Monte-Carlo samples otherwise. Gemini answers by letter.

| Model | MMLU | BLiMP | HellaSwag | PIQA | ARC-Easy | LAMBADA |
|---|---|---|---|---|---|---|
| TS-1M | 23.9 | 59.9 | 25.1 | 53.1 | 26.4 | 4.0 |
| TS-3M | 23.3 | 59.9 | 25.9 | 54.0 | 27.6 | 8.4 |
| TS-8M | 23.8 | 61.5 | 26.8 | 54.1 | 28.5 | 12.0 |
| TS-28M | 23.2 | 62.0 | 27.1 | 53.9 | 28.1 | 14.6 |
| TS-33M | 24.5 | 64.3 | 26.9 | 54.9 | 27.4 | 11.7 |
| TS-1Layer-21M | 25.1 | 63.1 | 26.8 | 55.4 | 28.1 | 6.2 |
| TS-2Layers-33M | 23.5 | 65.3 | 27.3 | 55.9 | 29.3 | 14.8 |
| Transformer | 23.3 | 62.4 | 26.7 | 55.3 | 27.8 | 11.8 |
| Looped | 23.2 | 63.4 | 26.6 | 55.6 | 27.3 | 11.5 |
| Diffusion | 25.0 | 55.8 | 26.1 | 51.1 | 27.5 | 8.3 |
| Looped diffusion | 23.7 | 56.2 | 26.3 | 50.7 | 27.4 | 8.4 |
| Gemini 3.8 Flash | 93.8 (subset) | – | – | – | – | – |
| Gemini 4 Argon | 94.6 (subset) | – | – | – | – | – |
