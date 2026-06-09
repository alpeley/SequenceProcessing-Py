#!/bin/bash
# Demo training run for Penn POS seq2seq Transformer.
# Usage: ./run_transformer_experiment.sh

set -e
cd "$(dirname "$0")"

python train_transformer.py \
  --train Datasets/postag-penn-train.txt \
  --dev Datasets/postag-penn-dev.txt \
  --test Datasets/postag-penn-test.txt \
  --max-train-sentences 100 \
  --max-eval-sentences 50 \
  --epoch 10 \
  --embedding-dim 7 \
  --num-heads 2 \
  --ffn-size 16 \
  --learning-rate 0.001 \
  --max-tokens 40 \
  --skip-autoregressive
