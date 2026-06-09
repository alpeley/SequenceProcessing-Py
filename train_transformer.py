#!/usr/bin/env python3
"""
Train the encoder-decoder Transformer on sequence datasets (e.g. Penn Treebank POS).

Example:
  python train_transformer.py --train Datasets/postag-penn-train.txt --dev Datasets/postag-penn-dev.txt
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Set, Tuple

from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Function.Sigmoid import Sigmoid
from ComputationalGraph.Function.Tanh import Tanh
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.AdamW import AdamW
from Math.Tensor import Tensor

from SequenceProcessing.Classification.Transformer import Transformer
from SequenceProcessing.Parameters.TransformerParameter import TransformerParameter
from SequenceProcessing.Sequence.TransformerCorpusDataset import TransformerCorpusDataset


class TeacherForcedMetrics(NamedTuple):
    """
    Teacher-forced evaluation metrics for one split.
    """
    fullAccuracy: float
    posOnlyAccuracy: float
    majorityBaselineFull: float
    majorityBaselinePosOnly: float
    goldDistribution: Dict[str, int]
    predictionDistribution: Dict[str, int]


def __specialTargetNames() -> Set[str]:
    """
    :return: Decoder target names excluded from POS-only accuracy.
    """
    return {"<S>", "</S>"}


def __indexToLabelName(dataset: TransformerCorpusDataset, index: int) -> str:
    """
    :param dataset: Dataset providing the label dictionary.
    :param index: Dictionary index.
    :return: Label name for the given index.
    """
    return dataset.getDictionary().getWordWithIndex(index).getName()


def computeTeacherForcedMetrics(dataset: TransformerCorpusDataset,
                                predictions: List[int],
                                gold_labels: List[int]) -> TeacherForcedMetrics:
    """
    Computes full and POS-only teacher-forced metrics plus majority baselines.

    :param dataset: Dataset used for label-name lookup.
    :param predictions: Predicted label indices.
    :param gold_labels: Gold label indices.
    :return: Teacher-forced metric bundle.
    """
    special_names = __specialTargetNames()
    full_correct = 0
    full_total = 0
    pos_correct = 0
    pos_total = 0
    gold_distribution: Counter[str] = Counter()
    prediction_distribution: Counter[str] = Counter()
    pos_gold_distribution: Counter[str] = Counter()

    for prediction_index, gold_index in zip(predictions, gold_labels):
        prediction_name = __indexToLabelName(dataset, prediction_index)
        gold_name = __indexToLabelName(dataset, gold_index)

        prediction_distribution[prediction_name] += 1
        gold_distribution[gold_name] += 1

        full_total += 1
        if prediction_index == gold_index:
            full_correct += 1

        if gold_name not in special_names:
            pos_total += 1
            pos_gold_distribution[gold_name] += 1
            if prediction_index == gold_index:
                pos_correct += 1

    full_accuracy = (full_correct + 0.0) / full_total if full_total > 0 else 0.0
    pos_only_accuracy = (pos_correct + 0.0) / pos_total if pos_total > 0 else 0.0

    majority_count_full = max(gold_distribution.values()) if gold_distribution else 0
    majority_baseline_full = (majority_count_full + 0.0) / full_total if full_total > 0 else 0.0

    majority_count_pos = max(pos_gold_distribution.values()) if pos_gold_distribution else 0
    majority_baseline_pos = (majority_count_pos + 0.0) / pos_total if pos_total > 0 else 0.0

    return TeacherForcedMetrics(
        fullAccuracy=full_accuracy,
        posOnlyAccuracy=pos_only_accuracy,
        majorityBaselineFull=majority_baseline_full,
        majorityBaselinePosOnly=majority_baseline_pos,
        goldDistribution=dict(gold_distribution),
        predictionDistribution=dict(prediction_distribution),
    )


def validateModelShape(embedding_dim: int, num_heads: int) -> int:
    """
    Transformer uses L = embedding_dim + 1; multi-head concat width must equal L.

    :param embedding_dim: Per-token embedding size without bias.
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


def buildTransformer(train_dataset: TransformerCorpusDataset,
                     epoch: int,
                     num_heads: int,
                     ffn_size: int,
                     learning_rate: float) -> Transformer:
    """
    Builds a Transformer model configured for the given dataset.

    :param train_dataset: Training corpus dataset wrapper.
    :param epoch: Training epochs.
    :param num_heads: Number of attention heads.
    :param ffn_size: Feed-forward hidden size.
    :param learning_rate: AdamW learning rate.
    :return: Untrained Transformer instance.
    """
    input_layers = [ffn_size]
    output_layers = [ffn_size]
    input_functions = [Tanh()]
    output_functions = [Sigmoid()]

    gamma_input = [1.0, 1.0]
    gamma_output = [1.0, 1.0, 1.0]
    beta_input = [0.0, 0.0]
    beta_output = [0.0, 0.0, 0.0]

    parameter = TransformerParameter(
        seed=1,
        epoch=epoch,
        optimizer=AdamW(learning_rate, 1.0, 0.99, 0.99, 1e-10, 0.01),
        initialization=RandomInitialization(),
        loss=CrossEntropyLoss(),
        word_embedding_length=train_dataset.getEmbeddingDim(),
        multi_head_attention_length=num_heads,
        vocabulary_length=train_dataset.getVocabularySize(),
        epsilon=1e-9,
        input_hidden_layers=input_layers,
        output_hidden_layers=output_layers,
        input_activation_functions=input_functions,
        output_activation_functions=output_functions,
        gamma_input_values=gamma_input,
        gamma_output_values=gamma_output,
        beta_input_values=beta_input,
        beta_output_values=beta_output,
    )

    return Transformer(parameter, train_dataset.getDictionary())


def printDictionaryMapping(dataset: TransformerCorpusDataset) -> None:
    """
    Prints the label dictionary indices used by the decoder.

    :param dataset: Training dataset with a built label dictionary.
    """
    label_to_index = dataset.getLabelToIndex()
    print("  label dictionary:")

    for label_name in sorted(label_to_index.keys()):
        print(f"    {label_to_index[label_name]:2d}: {label_name}")


def printFirstTensorDebug(dataset: TransformerCorpusDataset, tensor: Tensor) -> None:
    """
    Prints debug information for the first training tensor.

    :param dataset: Dataset that created the tensor.
    :param tensor: First sentence tensor.
    """
    summary = dataset.describeTensor(tensor)
    print("  first tensor debug:")
    print(f"    encoder token count: {summary['encoderTokenCount']}")
    print(f"    decoder step count: {summary['decoderStepCount']}")
    print(f"    gold label indices: {summary['goldLabelIndices']}")
    print(f"    gold label names: {summary['goldLabelNames']}")


def printGoldLabelDistribution(dataset: TransformerCorpusDataset,
                               tensors: List[Tensor]) -> None:
    """
    Prints the gold POS label distribution across tensors.

    :param dataset: Dataset used to build tensors.
    :param tensors: Tensor list to inspect.
    """
    counter: Counter[str] = Counter()

    for tensor in tensors:
        summary = dataset.describeTensor(tensor)
        counter.update(summary["goldLabelNames"])

    print(f"  training gold label distribution: {dict(counter)}")


def distributionToLabelNames(distribution: Dict[int, int],
                             dataset: TransformerCorpusDataset) -> Dict[str, int]:
    """
    Converts index-keyed distributions to label-name keys.

    :param distribution: Counts keyed by dictionary index.
    :param dataset: Dataset providing label names.
    :return: Counts keyed by label name.
    """
    dictionary = dataset.getDictionary()
    named_distribution = {}

    for index, count in distribution.items():
        label_name = dictionary.getWordWithIndex(index).getName()
        named_distribution[label_name] = count

    return named_distribution


def labelsToOneHot(class_labels: List[int], vocabulary_size: int) -> Tensor:
    """
    Converts integer class labels into one-hot tensor.

    :param class_labels: Gold class label indices.
    :param vocabulary_size: Number of output classes.
    :return: One-hot encoded label tensor.
    """
    values = []

    for class_label in class_labels:
        for j in range(vocabulary_size):
            values.append(1.0 if j == class_label else 0.0)

    return Tensor(values, (len(class_labels), vocabulary_size))


def collectPosLabelCounts(dataset: TransformerCorpusDataset,
                            tensors: List[Tensor]) -> Counter[str]:
    """
    Counts POS gold labels across tensors, excluding special decoder tokens.

    :param dataset: Dataset used to describe tensors.
    :param tensors: Training or evaluation tensors.
    :return: POS label counts.
    """
    special_names = __specialTargetNames()
    label_counts: Counter[str] = Counter()

    for tensor in tensors:
        summary = dataset.describeTensor(tensor)
        for label_name in summary["goldLabelNames"]:
            if label_name not in special_names:
                label_counts[label_name] += 1

    return label_counts


def balanceTrainTensors(dataset: TransformerCorpusDataset,
                        tensors: List[Tensor]) -> List[Tensor]:
    """
    Oversamples sentences that contain rare POS tags.

    Class-weighted CrossEntropyLoss is unavailable in ComputationalGraph, so
    inverse-frequency sentence duplication approximates class balancing.

    :param dataset: Training dataset.
    :param tensors: Original training tensors.
    :return: Balanced training tensor list.
    """
    if len(tensors) == 0:
        return tensors

    label_counts = collectPosLabelCounts(dataset, tensors)
    if not label_counts:
        return tensors

    max_count = max(label_counts.values())
    balanced_tensors: List[Tensor] = []

    for tensor in tensors:
        summary = dataset.describeTensor(tensor)
        pos_labels = {
            label_name
            for label_name in summary["goldLabelNames"]
            if label_name not in __specialTargetNames()
        }

        repeat_count = 1
        for label_name in pos_labels:
            inverse_weight = max_count / label_counts[label_name]
            repeat_count = max(repeat_count, int(round(inverse_weight)))

        for _ in range(repeat_count):
            balanced_tensors.append(tensor)

    return balanced_tensors


def evaluateTeacherForced(transformer: Transformer,
                          dataset: TransformerCorpusDataset,
                          tensors: List[Tensor],
                          embedding_dim: int,
                          vocabulary_size: int) -> TeacherForcedMetrics:
    """
    Evaluates the transformer with teacher forcing.

    During teacher-forced evaluation, the decoder receives the gold previous POS labels
    from the tensor, instead of its own previous predictions.

    :param transformer: Trained Transformer model.
    :param dataset: Evaluation dataset for label-name lookup.
    :param tensors: Evaluation tensors.
    :param embedding_dim: Embedding dimension without bias.
    :param vocabulary_size: Number of output classes.
    :return: Teacher-forced metrics including POS-only and majority baselines.
    """
    all_predictions: List[int] = []
    all_gold_labels: List[int] = []

    for instance in tensors:
        class_labels = transformer.createInputTensors(
            instance,
            transformer.input_nodes[0],
            transformer.input_nodes[1],
            embedding_dim
        )

        if len(class_labels) == 0:
            continue

        transformer.input_nodes[2].setValue(
            labelsToOneHot(class_labels, vocabulary_size)
        )

        predictions = transformer.predict()
        all_predictions.extend(int(prediction) for prediction in predictions)
        all_gold_labels.extend(int(gold) for gold in class_labels)

    return computeTeacherForcedMetrics(dataset, all_predictions, all_gold_labels)


def printEvaluationResult(sentence_count: int,
                          metrics: TeacherForcedMetrics,
                          autoregressive_accuracy: float,
                          skip_autoregressive: bool) -> None:
    """
    Prints autoregressive and teacher-forced evaluation results.

    :param sentence_count: Number of evaluated sentences.
    :param metrics: Teacher-forced metric bundle.
    :param autoregressive_accuracy: Autoregressive token accuracy from transformer.test().
    :param skip_autoregressive: Whether autoregressive eval was skipped.
    """
    print(f"  sentences: {sentence_count}")

    if skip_autoregressive:
        print("  autoregressive token accuracy: skipped (--skip-autoregressive)")
    else:
        print(f"  autoregressive token accuracy: {autoregressive_accuracy:.4f}")

    print(f"  teacher-forced token accuracy (full): {metrics.fullAccuracy:.4f}")
    print(f"  teacher-forced token accuracy (POS-only): {metrics.posOnlyAccuracy:.4f}")
    print(f"  majority baseline (full): {metrics.majorityBaselineFull:.4f}")
    print(f"  majority baseline (POS-only): {metrics.majorityBaselinePosOnly:.4f}")
    print(f"  gold distribution: {metrics.goldDistribution}")
    print(f"  prediction distribution: {metrics.predictionDistribution}")


def evaluateSplit(transformer: Transformer,
                  dataset: TransformerCorpusDataset,
                  tensors: List[Tensor],
                  embedding_dim: int,
                  vocabulary_size: int,
                  skip_autoregressive: bool) -> Tuple[TeacherForcedMetrics, float]:
    """
    Runs teacher-forced and optional autoregressive evaluation.

    :param transformer: Trained model.
    :param dataset: Evaluation dataset.
    :param tensors: Evaluation tensors.
    :param embedding_dim: Embedding dimension without bias.
    :param vocabulary_size: Output vocabulary size.
    :param skip_autoregressive: Skip slow autoregressive decoding.
    :return: Teacher-forced metrics and autoregressive accuracy.
    """
    metrics = evaluateTeacherForced(
        transformer,
        dataset,
        tensors,
        embedding_dim,
        vocabulary_size,
    )

    if skip_autoregressive:
        autoregressive_accuracy = 0.0
    else:
        autoregressive_accuracy = transformer.test(tensors).getAccuracy()

    printEvaluationResult(
        len(tensors),
        metrics,
        autoregressive_accuracy,
        skip_autoregressive,
    )

    return metrics, autoregressive_accuracy


def main() -> None:
    """
    Loads Penn POS splits, trains the Transformer, and reports accuracy.
    """
    parser = argparse.ArgumentParser(description="Train Transformer on SequenceCorpus datasets.")
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
        help="Test corpus file evaluated after training.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=7,
        help="Embedding size per token. L = embedding_dim + 1 must divide num_heads.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=3,
        help="Training epochs.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Attention head count.",
    )
    parser.add_argument(
        "--ffn-size",
        type=int,
        default=16,
        help="Feed-forward hidden size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
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
        help="Optional cap on training sentences. Default: all.",
    )
    parser.add_argument(
        "--max-eval-sentences",
        type=int,
        default=None,
        help="Optional cap on dev/test sentences. Default: all.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print dictionary mapping, label distributions, and first tensor debug info.",
    )
    parser.add_argument(
        "--skip-autoregressive",
        action="store_true",
        help="Skip slow autoregressive transformer.test() evaluation.",
    )
    parser.add_argument(
        "--no-balance-train",
        action="store_true",
        help="Disable inverse-frequency sentence oversampling.",
    )

    args = parser.parse_args()
    validateModelShape(args.embedding_dim, args.num_heads)

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    test_path = Path(args.test)

    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    print(f"Loading training data: {train_path}")
    train_dataset = TransformerCorpusDataset(
        str(train_path),
        embedding_dim=args.embedding_dim,
        max_tokens_per_sentence=args.max_tokens,
    )
    train_tensors = train_dataset.toTensorList(max_sentences=args.max_train_sentences)

    print(f"  {train_dataset}")
    print(f"  tensors: {len(train_tensors)}")

    if len(train_tensors) == 0:
        raise ValueError("No training tensors were created. Check corpus path and format.")

    dictionary = train_dataset.getDictionary()
    label_to_index = train_dataset.getLabelToIndex()

    print(f"  vocabulary size: {dictionary.size()}")
    print(f"  <S>={label_to_index['<S>']}  </S>={label_to_index['</S>']}")

    if args.debug:
        printDictionaryMapping(train_dataset)
        printGoldLabelDistribution(train_dataset, train_tensors)
        printFirstTensorDebug(train_dataset, train_tensors[0])

    if not args.no_balance_train:
        original_count = len(train_tensors)
        train_tensors = balanceTrainTensors(train_dataset, train_tensors)
        print(f"  balanced tensors: {len(train_tensors)} (from {original_count})")

    transformer = buildTransformer(
        train_dataset,
        epoch=args.epoch,
        num_heads=args.num_heads,
        ffn_size=args.ffn_size,
        learning_rate=args.learning_rate,
    )

    print(
        f"Model: Transformer(vocab={train_dataset.getVocabularySize()}, "
        f"L={args.embedding_dim + 1}, heads={args.num_heads})"
    )

    print("Training...")
    transformer.train(train_tensors)

    test_same_as_dev = dev_path.is_file() and test_path.is_file() and dev_path.resolve() == test_path.resolve()

    if dev_path.is_file():
        print(f"\nEvaluating dev: {dev_path}")
        dev_dataset = TransformerCorpusDataset(
            str(dev_path),
            embedding_dim=args.embedding_dim,
            dictionary=dictionary,
            max_tokens_per_sentence=args.max_tokens,
        )
        dev_tensors = dev_dataset.toTensorList(max_sentences=args.max_eval_sentences)

        evaluateSplit(
            transformer,
            dev_dataset,
            dev_tensors,
            args.embedding_dim,
            train_dataset.getVocabularySize(),
            args.skip_autoregressive,
        )
    else:
        print(f"Dev file not found, skipping: {dev_path}")

    if test_path.is_file() and not test_same_as_dev:
        print(f"\nEvaluating test: {test_path}")
        test_dataset = TransformerCorpusDataset(
            str(test_path),
            embedding_dim=args.embedding_dim,
            dictionary=dictionary,
            max_tokens_per_sentence=args.max_tokens,
        )
        test_tensors = test_dataset.toTensorList(max_sentences=args.max_eval_sentences)

        evaluateSplit(
            transformer,
            test_dataset,
            test_tensors,
            args.embedding_dim,
            train_dataset.getVocabularySize(),
            args.skip_autoregressive,
        )
    elif test_same_as_dev:
        print("\nTest split same as dev, skipping duplicate evaluation.")
    else:
        print(f"Test file not found, skipping: {test_path}")


if __name__ == "__main__":
    main()
