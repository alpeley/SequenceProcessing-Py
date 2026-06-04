#!/bin/bash

python train_gpt.py \
  --train Datasets/postag-penn-train.txt \
  --dev Datasets/postag-penn-dev.txt \
  --test Datasets/postag-penn-test.txt \
  --embedding-dim 7 \
  --num-heads 2 \
  --num-layers 6 \
  --ffn-size 32 \
  --epoch 5 \
  --learning-rate 0.005 \
  --max-train-sentences 100 \
  --max-eval-sentences 50 \
  --max-tokens 50