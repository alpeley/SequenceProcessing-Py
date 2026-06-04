#!/usr/bin/env python3
"""
Train GPT on fork sequence datasets (e.g. Penn Treebank POS).

Example:
  python train_gpt.py --train Datasets/postag-penn-train.txt --dev Datasets/postag-penn-dev.txt
"""

import argparse
from pathlib import Path

from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Function.Sigmoid import Sigmoid
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.AdamW import AdamW

from SequenceProcessing.Classification.GPT import GPT
from SequenceProcessing.Parameters.GPTParameter import GPTParameter
from SequenceProcessing.Sequence.GPTCorpusDataset import GPTCorpusDataset


def validateModelShape(embedding_dim: int, num_heads: int) -> int:
    """
    GPT uses L = embedding_dim + 1; multi-head concat width must equal L.

    :param embedding_dim: Per-token embedding size (without bias).
    :param num_heads: Attention head count.
    :return: Model dimension L.
    :raises ValueError: If L is not divisible by num_heads.
    """
    l_dim = embedding_dim + 1
    if l_dim % num_heads != 0:
        raise ValueError(
            f"(embedding_dim + 1) = {l_dim} must be divisible by num_heads = {num_heads}. "
            f"Example: --embedding-dim 7 --num-heads 2 gives L=8."
        )
    return l_dim


def layerNormParameterCount(num_layers: int) -> int:
    """
    GPT uses two layer norms per block plus one final layer norm.

    :param num_layers: Number of transformer blocks.
    :return: Required gamma/beta parameter count.
    """
    return 2 * num_layers + 1


def buildGpt(train_dataset: GPTCorpusDataset,
             epoch: int,
             num_layers: int,
             num_heads: int,
             ffn_size: int,
             learning_rate: float) -> GPT:
    """
    Builds a GPT model configured for the given dataset.

    :param train_dataset: Training corpus dataset wrapper.
    :param epoch: Training epochs.
    :param num_layers: Number of transformer blocks.
    :param num_heads: Number of attention heads.
    :param ffn_size: Feed-forward hidden size.
    :param learning_rate: AdamW learning rate.
    :return: Untrained GPT instance.
    """
    ln_count = layerNormParameterCount(num_layers)
    gamma_values = [1.0] * ln_count
    beta_values = [0.0] * ln_count

    parameter = GPTParameter(
        seed=1,
        epoch=epoch,
        optimizer=AdamW(learning_rate, 0.99, 0.99, 0.999, 1e-10, 0.1),
        initialization=RandomInitialization(),
        loss=CrossEntropyLoss(),
        word_embedding_length=train_dataset.getEmbeddingDim(),
        num_heads=num_heads,
        vocab_size=train_dataset.getVocabularySize(),
        num_layers=num_layers,
        epsilon=1e-9,
        ffn_size=ffn_size,
        activation_function=Sigmoid(),
        gamma_values=gamma_values,
        beta_values=beta_values,
    )

    return GPT(parameter)


def main() -> None:
    """
    Loads fork datasets, trains GPT, and reports dev accuracy.
    """
    parser = argparse.ArgumentParser(description="Train GPT on SequenceCorpus datasets.")
    parser.add_argument(
        "--train",
        default="Datasets/postag-penn-train.txt",
        help="Training corpus file.",
    )
    parser.add_argument(
        "--dev",
        default="Datasets/postag-penn-dev.txt",
        help="Development corpus file.",
    )
    parser.add_argument(
        "--test",
        default="Datasets/postag-penn-test.txt",
        help="Test corpus file (evaluated after training).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=7,
        help="Embedding size per token (L = embedding_dim + 1 must divide num_heads).",
    )
    parser.add_argument("--epoch", type=int, default=3, help="Training epochs.")
    parser.add_argument("--num-layers", type=int, default=1, help="Transformer block count.")
    parser.add_argument("--num-heads", type=int, default=2, help="Attention head count.")
    parser.add_argument("--ffn-size", type=int, default=32, help="Feed-forward hidden size.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Optimizer learning rate.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Skip sentences with more than this many tokens.",
    )
    parser.add_argument(
        "--max-train-sentences",
        type=int,
        default=None,
        help="Optional cap on training sentences (default: all).",
    )
    parser.add_argument(
        "--max-eval-sentences",
        type=int,
        default=None,
        help="Optional cap on dev/test sentences (default: all).",
    )
    args = parser.parse_args()
    validateModelShape(args.embedding_dim, args.num_heads)

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    test_path = Path(args.test)

    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    print(f"Loading training data: {train_path}")
    train_dataset = GPTCorpusDataset(
        str(train_path),
        embedding_dim=args.embedding_dim,
        max_tokens_per_sentence=args.max_tokens,
    )
    train_tensors = train_dataset.toTensorList(max_sentences=args.max_train_sentences)
    print(f"  {train_dataset}")
    print(f"  tensors: {len(train_tensors)}")

    if len(train_tensors) == 0:
        raise ValueError("No training tensors were created. Check corpus path and format.")

    label_map = train_dataset.getLabelToIndex()
    print(f"  labels ({len(label_map)}): {list(label_map.keys())}")

    gpt = buildGpt(
        train_dataset,
        epoch=args.epoch,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_size=args.ffn_size,
        learning_rate=args.learning_rate,
    )
    print(f"Model: {gpt}")

    print("Training...")
    gpt.train(train_tensors)

    if dev_path.is_file():
        print(f"\nEvaluating dev: {dev_path}")
        dev_dataset = GPTCorpusDataset(
            str(dev_path),
            embedding_dim=args.embedding_dim,
            label_to_index=label_map,
            max_tokens_per_sentence=args.max_tokens,
        )
        dev_tensors = dev_dataset.toTensorList(max_sentences=args.max_eval_sentences)
        dev_performance = gpt.test(dev_tensors)
        print(f"  sentences: {len(dev_tensors)}")
        print(f"  token accuracy: {dev_performance.getAccuracy():.4f}")
    else:
        print(f"Dev file not found, skipping: {dev_path}")

    if test_path.is_file():
        print(f"\nEvaluating test: {test_path}")
        test_dataset = GPTCorpusDataset(
            str(test_path),
            embedding_dim=args.embedding_dim,
            label_to_index=label_map,
            max_tokens_per_sentence=args.max_tokens,
        )
        test_tensors = test_dataset.toTensorList(max_sentences=args.max_eval_sentences)
        test_performance = gpt.test(test_tensors)
        print(f"  sentences: {len(test_tensors)}")
        print(f"  token accuracy: {test_performance.getAccuracy():.4f}")
    else:
        print(f"Test file not found, skipping: {test_path}")


if __name__ == "__main__":
    main()
