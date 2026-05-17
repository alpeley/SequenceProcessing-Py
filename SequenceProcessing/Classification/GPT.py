import math
import random
from typing import List, Tuple

from Classification.Performance.ClassificationPerformance import ClassificationPerformance
from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Function.Negation import Negation
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from Math.Tensor import Tensor

from SequenceProcessing.Functions.Inverse import Inverse
from SequenceProcessing.Functions.Mask import Mask
from SequenceProcessing.Functions.Mean import Mean
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.SquareRoot import SquareRoot
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Functions.Variance import Variance
from SequenceProcessing.Parameters.GPTParameter import GPTParameter


class GPT(ComputationalGraph):
    """
    GPT (decoder-only transformer) model implementation.
    Stacks masked self-attention blocks with no encoder component.
    """

    def __init__(self, parameter: NeuralNetworkParameter):
        """
        Constructor for GPT.

        :param parameter: Neural network parameter object (GPTParameter).
        """
        super().__init__(parameter)

    def __positionalEncoding(self, tensor: Tensor, word_embedding_length: int) -> Tensor:
        """
        Applies sinusoidal positional encoding to the input tensor.

        :param tensor: Input tensor of shape (T, word_embedding_length).
        :param word_embedding_length: Token embedding dimension (without bias column).
        :return: Positionally encoded tensor of the same shape.
        """
        values = []

        for i in range(tensor.getShape()[0]):
            for j in range(tensor.getShape()[1]):
                val = tensor.getValue((i, j))
                if j % 2 == 0:
                    values.append(val + math.sin((i + 1.0) / math.pow(10000, j / word_embedding_length)))
                else:
                    values.append(val + math.cos((i + 1.0) / math.pow(10000, (j - 1.0) / word_embedding_length)))

        return Tensor(values, tensor.getShape())

    def __createInputs(self, instance: Tensor, word_embedding_length: int) -> Tuple[Tensor, List[int]]:
        """
        Parses a flat training instance into a positionally encoded embedding tensor and class labels.

        Each token occupies (word_embedding_length + 1) consecutive values: the embedding followed by the
        target next-token index.

        :param instance: Flat input tensor of shape (T * (word_embedding_length + 1),).
        :param word_embedding_length: Token embedding dimension (without bias column).
        :return: Tuple of (positionally encoded embedding tensor of shape (T, word_embedding_length),
                           list of target class label indices of length T).
        """
        stride = word_embedding_length + 1
        num_tokens = instance.getShape()[0] // stride
        values = []
        class_labels = []

        for i in range(num_tokens):
            base = i * stride
            for j in range(word_embedding_length):
                values.append(instance.getValue((base + j,)))
            class_labels.append(int(instance.getValue((base + word_embedding_length,))))

        embedding_tensor = Tensor(values, (num_tokens, word_embedding_length))
        return self.__positionalEncoding(embedding_tensor, word_embedding_length), class_labels

    def __layerNormalization(self,
                             input_node: ComputationalNode,
                             parameter: GPTParameter,
                             ln_index: List[int]) -> ComputationalNode:
        """
        Applies layer normalization with learnable gamma and beta parameters.

        Uses a single scalar gamma and beta per normalization step, broadcast across the embedding dimension.

        :param input_node: Input computational node of effective shape (T, L).
        :param parameter: GPT parameter object.
        :param ln_index: Single-element list used as a mutable counter for gamma/beta indexing.
        :return: Normalized node of shape (T, L).
        """
        input_mean = self.addEdge(input_node, Mean())
        mean_neg = self.addEdge(input_mean, Negation())
        centered = self.addAdditionEdge(input_node, mean_neg, False)

        variance = self.addEdge(centered, Variance())
        root_variance = self.addEdge(variance, SquareRoot(parameter.getEpsilon()))
        inv_root_variance = self.addEdge(root_variance, Inverse())

        normalized = self.addEdge(centered, inv_root_variance, False, True)

        gamma_data = [parameter.getGammaValue(ln_index[0])] * parameter.getL()
        gamma_node = MultiplicationNode(True, False, Tensor(gamma_data, (1, parameter.getL())), True)
        scaled = self.addEdge(normalized, gamma_node)

        beta_data = [parameter.getBetaValue(ln_index[0])] * parameter.getL()
        beta_node = ComputationalNode(True, False, Tensor(beta_data, (1, parameter.getL())))
        ln_index[0] += 1

        return self.addAdditionEdge(scaled, beta_node, False)

    def __maskedMultiHeadAttention(self,
                                   input_node: ComputationalNode,
                                   parameter: GPTParameter,
                                   random_generator: random.Random) -> List[ComputationalNode]:
        """
        Builds causal (masked) multi-head self-attention outputs.

        Each token can only attend to itself and preceding tokens.

        :param input_node: Input node of effective shape (T, L).
        :param parameter: GPT parameter object.
        :param random_generator: Random number generator for weight initialization.
        :return: List of N per-head attention output nodes, each of shape (T, dk).
        """
        nodes = []

        for _ in range(parameter.getN()):
            wk = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            k = self.addEdge(input_node, wk)

            wq = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            q = self.addEdge(input_node, wq)

            wv = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            v = self.addEdge(input_node, wv)

            k_transpose = self.addEdge(k, Transpose())
            qk = self.addEdge(q, k_transpose, False, False)
            qk_scaled = self.addEdge(qk, MultiplyByConstant(1.0 / math.sqrt(parameter.getDk())))
            qk_masked = self.addEdge(qk_scaled, Mask())
            attn_weights = self.addEdge(qk_masked, Softmax())
            nodes.append(self.addEdge(attn_weights, v))

        return nodes

    def __feedForwardBlock(self,
                           current: ComputationalNode,
                           parameter: GPTParameter,
                           random_generator: random.Random) -> ComputationalNode:
        """
        Builds a two-layer position-wise feed-forward block: Linear → Activation → Linear.

        :param current: Input node of effective shape (T, L).
        :param parameter: GPT parameter object.
        :param random_generator: Random number generator for weight initialization.
        :return: Output node of shape (T, L).
        """
        w1 = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getL(), parameter.getFfnSize(), random_generator),
                (parameter.getL(), parameter.getFfnSize())
            )
        )
        hidden = self.addEdge(current, w1)
        hidden = self.addEdge(hidden, parameter.getActivationFunction(), True)

        w2 = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getFfnSize() + 1, parameter.getL(), random_generator),
                (parameter.getFfnSize() + 1, parameter.getL())
            )
        )
        return self.addEdge(hidden, w2)

    def train(self, train_set: List[Tensor]) -> None:
        """
        Builds the GPT computational graph and trains on the given dataset.

        Graph structure per block (pre-norm):
          x = x + Attention(LayerNorm(x))
          x = x + FFN(LayerNorm(x))
        Followed by a final LayerNorm, linear projection to vocab, and softmax.

        :param train_set: List of flat training tensors.
        """
        parameter = self.parameters
        random_generator = random.Random(parameter.getSeed())
        ln_index = [0]

        input_node = MultiplicationNode(False, True)
        self.input_nodes.append(input_node)

        current = input_node

        for _ in range(parameter.getNumLayers()):
            ln1 = self.__layerNormalization(current, parameter, ln_index)

            attn_nodes = self.__maskedMultiHeadAttention(ln1, parameter, random_generator)
            concat_attn = self.concatEdges(attn_nodes, 1)
            w_proj = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getL(), random_generator),
                    (parameter.getL(), parameter.getL())
                )
            )
            attn_out = self.addEdge(concat_attn, w_proj)
            current = self.addAdditionEdge(current, attn_out, False)

            ln2 = self.__layerNormalization(current, parameter, ln_index)
            ffn_out = self.__feedForwardBlock(ln2, parameter, random_generator)
            current = self.addAdditionEdge(current, ffn_out, False)

        final_ln = self.__layerNormalization(current, parameter, ln_index)

        w_out = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getL(), parameter.getV(), random_generator),
                (parameter.getL(), parameter.getV())
            )
        )
        logits = self.addEdge(final_ln, w_out)
        self.output_node = self.addEdge(logits, Softmax())

        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)
        self.addFunctionEdge([self.output_node, class_label_node], parameter.getLossFunction(), False)

        word_embedding_length = parameter.getL() - 1

        for _ in range(parameter.getEpoch()):
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                embedding_tensor, class_labels = self.__createInputs(instance, word_embedding_length)
                self.input_nodes[0].setValue(embedding_tensor)

                class_label_values = []
                for label in class_labels:
                    for j in range(parameter.getV()):
                        class_label_values.append(1.0 if j == label else 0.0)
                self.input_nodes[1].setValue(
                    Tensor(class_label_values, (len(class_labels), parameter.getV()))
                )

                self.forwardCalculation()
                self.backpropagation()

            parameter.getOptimizer().setLearningRate()

    def test(self, test_set: List[Tensor]) -> ClassificationPerformance:
        """
        Tests the GPT model using teacher forcing over the given test set.

        :param test_set: List of flat test tensors.
        :return: Classification performance based on next-token prediction accuracy.
        """
        count = 0
        total = 0
        word_embedding_length = self.parameters.getL() - 1

        for instance in test_set:
            embedding_tensor, gold_labels = self.__createInputs(instance, word_embedding_length)
            self.input_nodes[0].setValue(embedding_tensor)
            predictions = self.predict()

            for pred, gold in zip(predictions, gold_labels):
                if int(pred) == gold:
                    count += 1
                total += 1

        return ClassificationPerformance((count + 0.0) / total)

    def getOutputValue(self, computational_node: ComputationalNode) -> List[float]:
        """
        Extracts the argmax predicted class index for each token position from the output node.

        :param computational_node: Output node of shape (T, V).
        :return: List of predicted class indices, one per token position.
        """
        class_labels = []
        value = computational_node.getValue()

        for i in range(value.getShape()[0]):
            max_val = float("-inf")
            index = -1.0

            for j in range(value.getShape()[1]):
                current = value.getValue((i, j))
                if current > max_val:
                    max_val = current
                    index = float(j)

            class_labels.append(index)

        return class_labels

    def __repr__(self) -> str:
        """
        :return: String representation of GPT model.
        """
        parameter = self.parameters
        return (f"GPT(L={parameter.getL()}, N={parameter.getN()}, V={parameter.getV()}, "
                f"numLayers={parameter.getNumLayers()})")
