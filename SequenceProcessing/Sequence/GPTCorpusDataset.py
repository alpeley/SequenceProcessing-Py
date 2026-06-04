import hashlib
from typing import Dict, List, Optional

from Corpus.Sentence import Sentence
from Math.Tensor import Tensor
from Math.Vector import Vector

from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus


class GPTCorpusDataset:
    """
    Converts a SequenceCorpus text file into flat Tensor instances for GPT training.
    Each sentence becomes one Tensor: [emb_0, ..., emb_{d-1}, label_index] per token.
    """

    __corpus: SequenceCorpus
    __label_to_index: Dict[str, int]
    __embedding_dim: int
    __word_embeddings: Dict[str, List[float]]

    def __init__(self,
                 corpus_file: str,
                 embedding_dim: int = 8,
                 label_to_index: Optional[Dict[str, int]] = None,
                 max_tokens_per_sentence: Optional[int] = None):
        """
        :param corpus_file: Path to a labelled sequence file (e.g. Datasets/postag-penn-train.txt).
        :param embedding_dim: Number of embedding values per token (GPT uses L = embedding_dim + 1).
        :param label_to_index: Optional fixed label map from training data (for dev/test files).
        :param max_tokens_per_sentence: Skip sentences longer than this (avoids unstable long sequences).
        """
        self.__corpus = SequenceCorpus(corpus_file)
        self.__embedding_dim = embedding_dim
        self.__max_tokens_per_sentence = max_tokens_per_sentence
        self.__word_embeddings = {}

        if label_to_index is None:
            labels = sorted(self.__corpus.getClassLabels())
            self.__label_to_index = {label: index for index, label in enumerate(labels)}
        else:
            self.__label_to_index = label_to_index

    def getCorpus(self) -> SequenceCorpus:
        """
        :return: Underlying sequence corpus.
        """
        return self.__corpus

    def getLabelToIndex(self) -> Dict[str, int]:
        """
        :return: Class label to index mapping.
        """
        return self.__label_to_index

    def getVocabularySize(self) -> int:
        """
        :return: Number of distinct class labels.
        """
        return len(self.__label_to_index)

    def getEmbeddingDim(self) -> int:
        """
        :return: Word embedding dimension used in tensors.
        """
        return self.__embedding_dim

    def sentenceCount(self) -> int:
        """
        :return: Number of sentences in the corpus.
        """
        return self.__corpus.sentenceCount()

    def __wordEmbedding(self, word_name: str, vector: Vector) -> List[float]:
        """
        Builds a fixed-size embedding for a word.

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
            digest = hashlib.md5(word_name.encode("utf-8")).hexdigest()
            seed = int(digest, 16)
            values = []
            for i in range(self.__embedding_dim):
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                values.append(((seed % 2000) / 1000.0 - 1.0) * 0.1)
        elif len(values) < self.__embedding_dim:
            while len(values) < self.__embedding_dim:
                values.append(0.0)

        self.__word_embeddings[word_name] = values
        return values

    def sentenceToTensor(self, sentence: Sentence) -> Optional[Tensor]:
        """
        Converts one sentence to a flat Tensor instance.

        :param sentence: Labelled sentence from the corpus.
        :return: Flat tensor, or None if the sentence has no labelled tokens.
        """
        flat_values = []

        for i in range(sentence.wordCount()):
            word = sentence.getWord(i)
            if not isinstance(word, LabelledVectorizedWord):
                continue

            label = word.getClassLabel()
            if label not in self.__label_to_index:
                continue

            flat_values.extend(self.__wordEmbedding(word.getName(), word.getVector()))
            flat_values.append(float(self.__label_to_index[label]))

        if len(flat_values) == 0:
            return None

        return Tensor(flat_values, (len(flat_values),))

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
        return (f"GPTCorpusDataset(sentences={self.sentenceCount()}, "
                f"labels={self.getVocabularySize()}, embeddingDim={self.__embedding_dim})")
