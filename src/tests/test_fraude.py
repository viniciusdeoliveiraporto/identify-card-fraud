from .test_utils import get_data
from model.autoencoder import AutoencoderFraudDetector
import unittest

class TestModel(unittest.TestCase):

    def setUp(self):
        self.count = 256
        self.data_test = get_data(self.count)
        self.model = AutoencoderFraudDetector()
        self.model.train(epochs=10)  # Treina o modelo antes dos testes

    # Teste apenas com fraudes
    def test_only_frauds(self):
        rows = self.data_test["fraud"]
        
        for i, row in enumerate(rows):
            expected = int(row[-1])
            predicted = self.model.predict(row)
            self.assertEqual(expected, predicted, msg=f"OnlyFraud Case {i}")

    # Teste apenas com não fraudes
    def test_only_legit(self):
        rows = self.data_test["legit"]

        for i, row in enumerate(rows):
            expected = int(row[-1])
            predicted = self.model.predict(row)
            self.assertEqual(expected, predicted, msg=f"OnlyLegit Case {i}")

    # Teste com dados mistos (fraudes + não fraudes)
    def test_mixed_data(self):
        rows = self.data_test["fraud"] + self.data_test["legit"]

        for i, row in enumerate(rows):
            expected = int(row[-1])
            predicted = self.model.predict(row)
            self.assertEqual(expected, predicted, msg=f"Mixed Case {i}")
    
