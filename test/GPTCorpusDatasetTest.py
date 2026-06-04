import unittest
from pathlib import Path

from SequenceProcessing.Sequence.GPTCorpusDataset import GPTCorpusDataset


class GPTCorpusDatasetTest(unittest.TestCase):
    """
    Tests corpus to tensor conversion for GPT training.
    """

    def testPostagPennTrain(self):
        """
        Loads Penn POS training split and checks tensor layout.
        """
        train_file = Path("Datasets/postag-penn-train.txt")
        if not train_file.is_file():
            self.skipTest("Dataset file not available.")

        dataset = GPTCorpusDataset(str(train_file), embedding_dim=7)
        tensors = dataset.toTensorList(max_sentences=5)

        self.assertEqual(15, dataset.getVocabularySize())
        self.assertEqual(5, len(tensors))
        self.assertGreater(tensors[0].getShape()[0], 0)
        self.assertEqual(0, tensors[0].getShape()[0] % (dataset.getEmbeddingDim() + 1))

    def testSharedLabelMap(self):
        """
        Dev data must reuse the training label map.
        """
        train_file = Path("Datasets/postag-penn-train.txt")
        dev_file = Path("Datasets/postag-penn-dev.txt")
        if not train_file.is_file() or not dev_file.is_file():
            self.skipTest("Dataset files not available.")

        train_dataset = GPTCorpusDataset(str(train_file), embedding_dim=4)
        dev_dataset = GPTCorpusDataset(
            str(dev_file),
            embedding_dim=4,
            label_to_index=train_dataset.getLabelToIndex(),
        )

        self.assertEqual(train_dataset.getVocabularySize(), dev_dataset.getVocabularySize())
        self.assertEqual(train_dataset.getLabelToIndex(), dev_dataset.getLabelToIndex())


if __name__ == "__main__":
    unittest.main()
