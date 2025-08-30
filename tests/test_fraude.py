from .test_utils import get_data
from model.model import Model
import unittest

class TestModel(unittest.TestCase):
   
    def setUp(self):
        self.count = 100
        self.data_test = get_data(self.count)
        self.model = Model()

    def test_model(self):
        rows = self.data_test["fraud"] + self.data_test["legit"]

        for i in range(len(rows)):
            response = int(rows[i][-1])
            model_response = self.model.predict(rows[i])
            self.assertEqual(response, model_response, msg=f"Test Case {i}")
    
