#!/usr/bin/env bash
# Zero-shot benchmarks for every local model -> results/harness/<model>/
#   bash eval/run_harness.sh ts                       # TinyStories checkpoints (TPU VM's CPU)
#   bash eval/run_harness.sh ours ~/project/ckpt       # our four JAX models (TPU)
set -e
TASKS=${TASKS:-mmlu,blimp,hellaswag,piqa,arc_easy,lambada_openai}
PY=${PY:-python}

if [ "$1" = ts ]; then
  for m in 1M 3M 8M 28M 33M 1Layer-21M 2Layers-33M; do
    $PY -m lm_eval --model hf \
      --model_args pretrained=roneneldan/TinyStories-$m,tokenizer=EleutherAI/gpt-neo-125M \
      --tasks $TASKS --device ${DEVICE:-cpu} --batch_size ${BS:-64} \
      --output_path results/harness/ts-$m
  done
elif [ "$1" = ours ]; then
  for m in baseline looped mdlm looped_mdlm; do
    $PY -m eval.harness_lm "$2/$m" --tasks $TASKS --out results/harness/$m
  done
fi
