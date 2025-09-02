from src.model.autoencoder import AutoencoderFraudDetector
from src.data.split_dataset import split_train_test
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
        failures = 0
        for x, y_true in zip(test_set, test_labels):
            pred = self.model.predict(x)  
            if (pred != y_true):
                failures+=1
        
        """Se a quantidade de erros do modelo ultrapassar 20% do total de testes, FALHA"""
        if (failures > int(len(frauds)*0.2)):
            self.fail(f"De {len(frauds)} testes {failures} asserts falharam:\n")

    def test_model_only_legit(self):
        """Todos os testes não são fraudes"""
        
        normals = self.ds_test[self.labels_test == 0]
        test_set = [row.tolist() for row in normals]
        test_labels =  [0]*len(normals)

        #testa linha a linha
        failures = 0
        for x, y_true in zip(test_set, test_labels):
            pred = self.model.predict(x)  
            if (pred != y_true):
                failures+=1
        
        """Se a quantidade de erros do modelo ultrapassar 20% do total de testes, FALHA"""
        if (failures > int(len(normals)*0.2)):
            self.fail(f"De {len(normals)} testes {failures} asserts falharam:\n") 

    def test_model_ninety_fraud(self):
        """90% dos testes não são fraudes e 10% são fraudes"""

        frauds = self.ds_test[self.labels_test == 1]
        normals = self.ds_test[self.labels_test == 0]

        "O número de fraudes é muito menor que o número de transações normais então limitamos o tamanho total do teste para manter a proporção de 9:1"
        total_test = min(len(frauds), len(normals))  
        if total_test == 0:
            self.fail("Não há fraudes ou normais suficientes para este teste.")
        
        n_fraud = max(1, int(total_test * 0.1))      # 10% fraudes
        n_normal = min(len(normals), int(n_fraud * 9)) # 90% normais

     
        test_set = [row.tolist() for row in frauds[:n_fraud]] + [row.tolist() for row in normals[:n_normal]]
        test_labels = [1]*n_fraud + [0]*n_normal

        #testa linha a linha
        failures = 0
        for x, y_true in zip(test_set, test_labels):
            pred = self.model.predict(x)
            if (pred != y_true):
                failures+=1

        """Se a quantidade de erros do modelo ultrapassar 20% do total de testes, FALHA"""
        if (failures > int(len(test_labels)*0.2)):
            self.fail(f"De {len(test_labels)} testes {failures} asserts falharam:\n") 

    
    def test_model_half_fraud(self):
        """50% dos testes são fraudes e os outros 50% não são fraudes"""

        frauds = self.ds_test[self.labels_test == 1]
        normals = self.ds_test[self.labels_test == 0]

        "O número de fraudes é muito menor que o número de transações normais então limitamos o tamanho total do teste para manter a proporção de 5:5"
        total_test = min(len(frauds), len(normals))
        test_set = [row.tolist() for row in frauds[:total_test]] + [row.tolist() for row in normals[:total_test]]
        test_labels = [1]*total_test + [0]*total_test

        failures = 0
        for x, y_true in zip(test_set, test_labels):
            pred = self.model.predict(x)  
            if (pred != y_true):
                failures+=1
        
        """Se a quantidade de erros do modelo ultrapassar 20% do total de testes, FALHA"""
        if (failures > int(len(test_labels)*0.2)):
            self.fail(f"De {len(test_labels)} testes {failures} asserts falharam:\n") 
    

if __name__ == "__main__":
    unittest.main()
    
