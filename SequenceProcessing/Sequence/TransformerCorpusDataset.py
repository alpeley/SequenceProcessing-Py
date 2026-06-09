import hashlib
import math
from functools import cmp_to_key
from typing import Dict, List, Optional

from Corpus.Sentence import Sentence
from Dictionary.VectorizedDictionary import VectorizedDictionary
from Dictionary.VectorizedWord import VectorizedWord
from Math.Tensor import Tensor
from Math.Vector import Vector

from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus


class TransformerCorpusDataset:
    """
    Converts a SequenceCorpus file into flat Tensor instances for Transformer seq2seq training.

    Each sentence becomes one Tensor:
    encoder word embeddings, float("inf"), then decoder chunks of
    [label embedding values + target label index].
    """

    __corpus: SequenceCorpus
    __dictionary: VectorizedDictionary
    __label_to_index: Dict[str, int]
    __embedding_dim: int
    __word_embeddings: Dict[str, List[float]]
    __label_embeddings: Dict[str, List[float]]
    __max_tokens_per_sentence: Optional[int]

    def __init__(self,
                 corpus_file: str,
                 embedding_dim: int = 8,
                 dictionary: Optional[VectorizedDictionary] = None,
                 max_tokens_per_sentence: Optional[int] = None):
        """
        :param corpus_file: Path to a labelled sequence file (e.g. Datasets/postag-penn-train.txt).
        :param embedding_dim: Number of embedding values per token (Transformer uses L = embedding_dim + 1).
        :param dictionary: Optional fixed label dictionary from training data (for dev/test files).
        :param max_tokens_per_sentence: Skip sentences longer than this (avoids unstable long sequences).
        """
        self.__corpus = SequenceCorpus(corpus_file)
        self.__embedding_dim = embedding_dim
        self.__max_tokens_per_sentence = max_tokens_per_sentence
        self.__word_embeddings = {}
        self.__label_embeddings = {}

        if dictionary is None:
            self.__dictionary = self.buildDictionary(
                sorted(self.__corpus.getClassLabels()),
                embedding_dim,
            )
        else:
            self.__dictionary = dictionary

        self.__label_to_index = self.__buildLabelToIndex(self.__dictionary)

    @staticmethod
    def buildDictionary(class_labels: List[str],
                        embedding_dim: int) -> VectorizedDictionary:
        """
        Builds a VectorizedDictionary with "<S>", "</S>", and POS label tokens.

        :param class_labels: Sorted class labels from the training corpus.
        :param embedding_dim: Embedding dimension for each dictionary token.
        :return: Dictionary containing special and label tokens.
        """
        dictionary = VectorizedDictionary()
        token_names = ["<S>", "</S>"] + list(class_labels)

        for token_name in token_names:
            dictionary.addWord(VectorizedWord(token_name, Vector([0.0])))

        dictionary.words.sort(key=cmp_to_key(dictionary.comparator))
        vocabulary_size = dictionary.size()

        for index in range(vocabulary_size):
            word = dictionary.getWordWithIndex(index)
            vector_values = TransformerCorpusDataset.__labelIndexEmbedding(
                index,
                embedding_dim,
                vocabulary_size,
            )
            dictionary.words[index] = VectorizedWord(word.getName(), Vector(vector_values))

        return dictionary

    @staticmethod
    def __labelIndexEmbedding(label_index: int,
                              embedding_dim: int,
                              vocabulary_size: int) -> List[float]:
        """
        Builds a distinct embedding for each dictionary index.

        Index-based sinusoidal features separate POS labels more clearly than
        hash embeddings, which helps the decoder avoid collapsing to one tag.

        :param label_index: Dictionary index of the label token.
        :param embedding_dim: Target embedding length.
        :param vocabulary_size: Total dictionary size.
        :return: Embedding values.
        """
        values = []
        scale = 0.25
        base = (label_index + 1.0) / max(vocabulary_size, 1)

        for dimension in range(embedding_dim):
            frequency = base * (dimension + 1.0)
            if dimension % 2 == 0:
                values.append(scale * math.sin(frequency * math.pi))
            else:
                values.append(scale * math.cos(frequency * math.pi))

        return values

    @staticmethod
    def __buildLabelToIndex(dictionary: VectorizedDictionary) -> Dict[str, int]:
        """
        Builds a reliable label-name to index map from sorted dictionary order.

        Dictionary.getWordIndex() can return incorrect indices for tokens such as
        NOUN or </S>, so indices must be taken from getWordWithIndex().

        :param dictionary: Sorted label dictionary.
        :return: Label name to index mapping.
        """
        label_to_index = {}

        for i in range(dictionary.size()):
            label_name = dictionary.getWordWithIndex(i).getName()
            label_to_index[label_name] = i

        return label_to_index

    @staticmethod
    def __hashEmbedding(token_name: str, embedding_dim: int) -> List[float]:
        """
        Builds a deterministic hash-based embedding for a token name.

        :param token_name: Token surface form.
        :param embedding_dim: Target embedding length.
        :return: Embedding values.
        """
        digest = hashlib.md5(token_name.encode("utf-8")).hexdigest()
        seed = int(digest, 16)
        values = []

        for _ in range(embedding_dim):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            values.append(((seed % 2000) / 1000.0 - 1.0) * 0.1)

        return values

    def getCorpus(self) -> SequenceCorpus:
        """
        :return: Underlying sequence corpus.
        """
        return self.__corpus

    def getDictionary(self) -> VectorizedDictionary:
        """
        :return: Label dictionary used by the decoder.
        """
        return self.__dictionary

    def getLabelToIndex(self) -> Dict[str, int]:
        """
        :return: Label name to dictionary index mapping.
        """
        return self.__label_to_index

    def getVocabularySize(self) -> int:
        """
        :return: Number of tokens in the label dictionary (includes "<S>" and "</S>").
        """
        return self.__dictionary.size()

    def getEmbeddingDim(self) -> int:
        """
        :return: Word and label embedding dimension used in tensors.
        """
        return self.__embedding_dim

    def sentenceCount(self) -> int:
        """
        :return: Number of sentences in the corpus.
        """
        return self.__corpus.sentenceCount()

    def __wordEmbedding(self, word_name: str, vector: Vector) -> List[float]:
        """
        Builds a fixed-size embedding for an input word.

        Uses corpus vector values when non-zero; otherwise a deterministic hash embedding.

        :param word_name: Surface form of the word.
        :param vector: Vector from the corpus (often 300-dim zeros).
        :return: Embedding of length embedding_dim.
        """
        if word_name in self.__word_embeddings:
            return self.__word_embeddings[word_name]

        values = []
        has_signal = False
        limit = min(self.__embedding_dim, vector.size())

        for i in range(limit):
            val = vector.getValue(i)
            values.append(val)
            if val != 0.0:
                has_signal = True

        if not has_signal:
            values = self.__hashEmbedding(word_name, self.__embedding_dim)
        elif len(values) < self.__embedding_dim:
            while len(values) < self.__embedding_dim:
                values.append(0.0)

        self.__word_embeddings[word_name] = values
        return values

    def __labelEmbedding(self, label_name: str) -> List[float]:
        """
        Returns the decoder embedding for a label token from the dictionary.

        :param label_name: Label token name (e.g. "DET", "<S>").
        :return: Embedding of length embedding_dim.
        """
        if label_name in self.__label_embeddings:
            return self.__label_embeddings[label_name]

        word = self.__dictionary.getWord(label_name)
        if word is None:
            values = self.__hashEmbedding(label_name, self.__embedding_dim)
        else:
            vector = word.getVector()
            values = [vector.getValue(i) for i in range(vector.size())]

        self.__label_embeddings[label_name] = values
        return values

    def sentenceToTensor(self, sentence: Sentence) -> Optional[Tensor]:
        """
        Converts one sentence to a flat seq2seq Tensor instance.

        Layout:
        emb(word_0)...emb(word_n), inf,
        emb(<S>) target label_0, emb(label_0) target label_1, ..., emb(label_n) target </S>.

        :param sentence: Labelled sentence from the corpus.
        :return: Flat tensor, or None if the sentence has no usable labelled tokens.
        """
        word_entries: List[LabelledVectorizedWord] = []
        label_names: List[str] = []

        for i in range(sentence.wordCount()):
            word = sentence.getWord(i)
            if not isinstance(word, LabelledVectorizedWord):
                continue

            label = word.getClassLabel()
            if label not in self.__label_to_index:
                continue

            word_entries.append(word)
            label_names.append(label)

        if len(word_entries) == 0:
            return None

        flat_values: List[float] = []

        for word in word_entries:
            flat_values.extend(self.__wordEmbedding(word.getName(), word.getVector()))

        flat_values.append(float("inf"))

        decoder_tokens = ["<S>"] + label_names
        target_labels = label_names + ["</S>"]

        for decoder_token, target_label in zip(decoder_tokens, target_labels):
            flat_values.extend(self.__labelEmbedding(decoder_token))
            flat_values.append(float(self.__label_to_index[target_label]))

        return Tensor(flat_values, (len(flat_values),))

    def describeTensor(self, tensor: Tensor) -> Dict[str, object]:
        """
        Returns a debug summary of one seq2seq tensor layout.

        :param tensor: Sentence tensor produced by sentenceToTensor().
        :return: Debug dictionary with encoder/decoder counts and gold labels.
        """
        flat_values = tensor.getData()
        separator_index = flat_values.index(float("inf"))
        encoder_token_count = separator_index // self.__embedding_dim
        decoder_flat_length = len(flat_values) - separator_index - 1
        decoder_step_count = decoder_flat_length // (self.__embedding_dim + 1)

        gold_indices = []
        decoder_values = flat_values[separator_index + 1:]

        for step in range(decoder_step_count):
            target_index = int(
                decoder_values[step * (self.__embedding_dim + 1) + self.__embedding_dim]
            )
            gold_indices.append(target_index)

        gold_names = [
            self.__dictionary.getWordWithIndex(index).getName()
            for index in gold_indices
        ]

        return {
            "encoderTokenCount": encoder_token_count,
            "decoderStepCount": decoder_step_count,
            "goldLabelIndices": gold_indices,
            "goldLabelNames": gold_names,
        }

    def toTensorList(self, max_sentences: Optional[int] = None) -> List[Tensor]:
        """
        Converts the corpus into a list of sentence tensors.

        :param max_sentences: Optional cap on number of sentences (for faster experiments).
        :return: List of Tensor instances.
        """
        tensors = []
        limit = self.__corpus.sentenceCount()

        if max_sentences is not None:
            limit = min(limit, max_sentences)

        for i in range(limit):
            sentence = self.__corpus.getSentence(i)
            if self.__max_tokens_per_sentence is not None:
                if sentence.wordCount() > self.__max_tokens_per_sentence:
                    continue
            tensor = self.sentenceToTensor(sentence)
            if tensor is not None:
                tensors.append(tensor)

        return tensors

    def __repr__(self) -> str:
        """
        :return: String representation.
        """
        return (f"TransformerCorpusDataset(sentences={self.sentenceCount()}, "
                f"vocabulary={self.getVocabularySize()}, embeddingDim={self.__embedding_dim})")
