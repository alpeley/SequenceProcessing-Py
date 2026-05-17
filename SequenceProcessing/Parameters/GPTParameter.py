from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class GPTParameter(NeuralNetworkParameter):
    """
    Parameter class for GPT (decoder-only transformer) model.
    """

    __L: int
    __N: int
    __V: int
    __numLayers: int
    __epsilon: float
    __ffnSize: int
    __activationFunction: Function
    __gammaValues: List[float]
    __betaValues: List[float]

    def __init__(self,
                 seed: int,
                 epoch: int,
                 optimizer: Optimizer,
                 initialization: Initialization,
                 loss: Function,
                 word_embedding_length: int,
                 num_heads: int,
                 vocab_size: int,
                 num_layers: int,
                 epsilon: float,
                 ffn_size: int,
                 activation_function: Function,
                 gamma_values: List[float],
                 beta_values: List[float]):
        """
        Constructor for GPTParameter.

        :param seed: Random seed.
        :param epoch: Number of epochs.
        :param optimizer: Optimizer.
        :param initialization: Weight initialization.
        :param loss: Loss function.
        :param word_embedding_length: Embedding dimension.
        :param num_heads: Number of attention heads.
        :param vocab_size: Vocabulary size.
        :param num_layers: Number of transformer blocks.
        :param epsilon: Layer norm stability constant.
        :param ffn_size: Feed-forward hidden size.
        :param activation_function: FFN activation function.
        :param gamma_values: Gamma values for each layer norm.
        :param beta_values: Beta values for each layer norm.
        """
        super().__init__(seed, epoch, optimizer, initialization, loss, 0.0, 1)

        self.__L = word_embedding_length + 1
        self.__N = num_heads
        self.__V = vocab_size
        self.__numLayers = num_layers
        self.__epsilon = epsilon
        self.__ffnSize = ffn_size
        self.__activationFunction = activation_function
        self.__gammaValues = gamma_values
        self.__betaValues = beta_values

    def getL(self) -> int:
        """
        :return: Embedding dimension (word_embedding_length + 1).
        """
        return self.__L

    def getN(self) -> int:
        """
        :return: Number of attention heads.
        """
        return self.__N

    def getV(self) -> int:
        """
        :return: Vocabulary size.
        """
        return self.__V

    def setV(self, vocab_size: int) -> None:
        """
        :param vocab_size: New vocabulary size.
        """
        self.__V = vocab_size

    def getNumLayers(self) -> int:
        """
        :return: Number of transformer blocks.
        """
        return self.__numLayers

    def getEpsilon(self) -> float:
        """
        :return: Epsilon for layer normalization.
        """
        return self.__epsilon

    def getDk(self) -> int:
        """
        :return: Attention head dimension (L // N).
        """
        return self.__L // self.__N

    def getFfnSize(self) -> int:
        """
        :return: Feed-forward hidden size.
        """
        return self.__ffnSize

    def getActivationFunction(self) -> Function:
        """
        :return: FFN activation function.
        """
        return self.__activationFunction

    def setActivationFunction(self, activation_function: Function) -> None:
        """
        :param activation_function: New activation function.
        """
        self.__activationFunction = activation_function

    def getGammaValue(self, index: int) -> float:
        """
        :param index: Layer norm index.
        :return: Gamma value.
        """
        return self.__gammaValues[index]

    def getBetaValue(self, index: int) -> float:
        """
        :param index: Layer norm index.
        :return: Beta value.
        """
        return self.__betaValues[index]

    def getGammaValues(self) -> List[float]:
        """
        :return: All gamma values.
        """
        return self.__gammaValues

    def setGammaValues(self, gamma_values: List[float]) -> None:
        """
        :param gamma_values: New gamma values.
        """
        self.__gammaValues = gamma_values

    def getBetaValues(self) -> List[float]:
        """
        :return: All beta values.
        """
        return self.__betaValues

    def setBetaValues(self, beta_values: List[float]) -> None:
        """
        :param beta_values: New beta values.
        """
        self.__betaValues = beta_values

    def __repr__(self) -> str:
        """
        :return: String representation of GPTParameter.
        """
        return (f"GPTParameter(L={self.__L}, N={self.__N}, V={self.__V}, "
                f"numLayers={self.__numLayers}, ffnSize={self.__ffnSize})")
