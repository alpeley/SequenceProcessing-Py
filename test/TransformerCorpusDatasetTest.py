import unittest
from pathlib import Path

from SequenceProcessing.Sequence.TransformerCorpusDataset import TransformerCorpusDataset


class TransformerCorpusDatasetTest(unittest.TestCase):
    """
    Tests corpus to tensor conversion for Transformer seq2seq training.
    """

    def testPostagPennTrain(self):
        """
        Loads Penn POS training split and checks seq2seq tensor layout.
        """
        train_file = Path("Datasets/postag-penn-train.txt")
        if not train_file.is_file():
            self.skipTest("Dataset file not available.")

        embedding_dim = 7
        dataset = TransformerCorpusDataset(str(train_file), embedding_dim=embedding_dim)
        tensors = dataset.toTensorList(max_sentences=5)

        self.assertEqual(17, dataset.getVocabularySize())
        self.assertEqual(5, len(tensors))
        self.assertGreater(tensors[0].getShape()[0], 0)

        flat = tensors[0].getData()
        self.assertIn(float("inf"), flat)

        inf_index = flat.index(float("inf"))
        encoder_length = inf_index
        decoder_length = len(flat) - inf_index - 1

        self.assertEqual(0, encoder_length % embedding_dim)
        self.assertEqual(0, decoder_length % (embedding_dim + 1))

        dictionary = dataset.getDictionary()
        label_to_index = dataset.getLabelToIndex()
        self.assertIsNotNone(dictionary.getWordWithIndex(label_to_index["<S>"]))
        self.assertIsNotNone(dictionary.getWordWithIndex(label_to_index["</S>"]))
        self.assertEqual("<S>", dictionary.getWordWithIndex(label_to_index["<S>"]).getName())
        self.assertEqual("</S>", dictionary.getWordWithIndex(label_to_index["</S>"]).getName())

    def testTargetLabelIndices(self):
        """
        Gold decoder targets must match the sentence POS labels, not collapsed indices.
        """
        train_file = Path("Datasets/postag-penn-train.txt")
        if not train_file.is_file():
            self.skipTest("Dataset file not available.")

        dataset = TransformerCorpusDataset(str(train_file), embedding_dim=7)
        sentence = dataset.getCorpus().getSentence(0)
        tensor = dataset.sentenceToTensor(sentence)
        summary = dataset.describeTensor(tensor)

        expected_labels = [
            sentence.getWord(i).getClassLabel()
            for i in range(sentence.wordCount())
        ] + ["</S>"]

        self.assertEqual(expected_labels, summary["goldLabelNames"])
        self.assertEqual(1, summary["goldLabelNames"].count("</S>"))

    def testSharedDictionary(self):
        """
        Dev data must reuse the training label dictionary.
        """
        train_file = Path("Datasets/postag-penn-train.txt")
        dev_file = Path("Datasets/postag-penn-dev.txt")
        if not train_file.is_file() or not dev_file.is_file():
            self.skipTest("Dataset files not available.")

        train_dataset = TransformerCorpusDataset(str(train_file), embedding_dim=4)
        dev_dataset = TransformerCorpusDataset(
            str(dev_file),
            embedding_dim=4,
            dictionary=train_dataset.getDictionary(),
        )

        self.assertEqual(train_dataset.getVocabularySize(), dev_dataset.getVocabularySize())
        self.assertEqual(
            train_dataset.getLabelToIndex(),
            dev_dataset.getLabelToIndex(),
        )

    def testSeq2SeqStructure(self):
        """
        Verifies encoder/decoder chunking for a single sentence.
        """
        train_file = Path("Datasets/postag-penn-train.txt")
        if not train_file.is_file():
            self.skipTest("Dataset file not available.")

        embedding_dim = 3
        dataset = TransformerCorpusDataset(str(train_file), embedding_dim=embedding_dim)
        sentence = dataset.getCorpus().getSentence(0)
        tensor = dataset.sentenceToTensor(sentence)

        self.assertIsNotNone(tensor)
        flat = tensor.getData()

        token_count = 0
        for i in range(sentence.wordCount()):
            word = sentence.getWord(i)
            if hasattr(word, "getClassLabel"):
                token_count += 1

        inf_index = flat.index(float("inf"))
        expected_encoder_length = token_count * embedding_dim
        expected_decoder_length = (token_count + 1) * (embedding_dim + 1)

        self.assertEqual(expected_encoder_length, inf_index)
        self.assertEqual(expected_decoder_length, len(flat) - inf_index - 1)


if __name__ == "__main__":
    unittest.main()
