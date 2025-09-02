from ..model.autoencoder import AutoencoderFraudDetector
from ..data.split_dataset import split_train_test
import unittest


class TestModel(unittest.TestCase):
   
    def setUp(self):
        self.ds_train, self.ds_test, self.labels_test = split_train_test()
        self.model = AutoencoderFraudDetector()
        self.model.train(epochs=20)

    def test_model_only_fraud(self):
        """Todos os testes são fraudes"""
        frauds = self.ds_test[self.labels_test == 1]
        test_set = [row.tolist() for row in frauds]
        test_labels =  [1]*len(frauds)

        #testa linha a linha
        i = 0
        for x, y_true in zip(test_set, test_labels):
            pred = self.model.predict(x)  
            self.assertEqual(pred, y_true, msg=f" TEST {i} ")  
            i+=1

    # def test_model_only_legit(self):
    #     """Todos os testes não são fraudes"""
        
    #     normals = self.ds_test[self.labels_test == 0]
    #     test_set = [row.tolist() for row in normals]
    #     test_labels =  [1]*len(normals)

    #     #testa linha a linha
    #     for x, y_true in zip(test_set, test_labels):
    #         pred = self.model.predict(x)  
    #         self.assertEqual(pred, y_true)  

    # def test_model_ninety_fraud(self):
    #     """90% dos testes não são fraudes e 10% são fraudes"""

    #     frauds = self.ds_test[self.labels_test == 1]
    #     normals = self.ds_test[self.labels_test == 0]

    #     total_test = min(len(frauds), len(normals))  
    #     n_fraud = max(1, int(total_test * 0.1))      # 10% fraudes
    #     n_normal = min(len(normals), int(n_fraud * 9)) # 90% normais

    #     test_set = [row.tolist() for row in frauds[:n_fraud]] + [row.tolist() for row in normals[:n_normal]]
    #     test_labels = [1]*n_fraud + [0]*n_normal

    #     #testa linha a linha
    #     for x, y_true in zip(test_set, test_labels):
    #         pred = self.model.predict(x)
    #         self.assertEqual(pred, y_true)

    
    # def test_model_half_fraud(self):
    #     """50% dos testes são fraudes e os outros 50% não são fraudes"""

    #     # frauds = self.ds_test[self.labels_test == 1]
    #     # normals = self.ds_test[self.labels_test == 0]
    #     # size = min(len(frauds), len(normals))
    #     # test_set = [row.tolist() for row in frauds[:size]] + [row.tolist() for row in normals[:size]]
    #     # test_labels = [1]*size + [0]*size

    #     # for x, y_true in zip(test_set, test_labels):
    #     #     pred = self.model.predict(x)  
    #     #     self.assertEqual(pred, y_true)        
    #     pass
    

if __name__ == "__main__":
    unittest.main()