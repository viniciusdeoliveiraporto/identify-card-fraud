
"""
Testes para AutoencoderFraudDetector.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.model.autoencoder import AutoencoderFraudDetector
from src.data.split_dataset import split_train_test
from src.utils.dataset_utils import load_dataset


class TestModel(unittest.TestCase):
    """Testa predição, treino, salvamento e avaliação do modelo de fraude."""

    @classmethod
    def setUpClass(cls):
        cls.ds_train, cls.ds_val, cls.ds_test, cls.labels_test = split_train_test()
        cls.model = AutoencoderFraudDetector()
        cls.model.train(epochs=10)

    def _run_test_set(self, test_set, test_labels, tolerance=0.2):  # pragma: no cover
        failures = 0
        for x, y_true in zip(test_set, test_labels):
            pred = self.model.predict(x)
            if pred != y_true:
                failures += 1

        #Se a quantidade de erros do modelo ultrapassar 20% do total de testes, FALHA"""
        if failures > int(len(test_labels) * tolerance):
            self.fail(f"De {len(test_labels)} testes {failures} falharam.")

    ##---------- Testes de predição ----------##
    def test_only_fraud(self):
        """Todos os testes são fraudes"""
        frauds = self.ds_test[self.labels_test == 1]
        if len(frauds) == 0: # pragma: no cover
            self.skipTest("Não há fraudes suficientes para o teste")

        test_set = [row.tolist() for row in frauds]
        test_labels =  [1]*len(frauds)
        self._run_test_set(test_set, test_labels)


    def test_model_only_legit(self):
        """Todos os testes não são fraudes"""

        normals = self.ds_test[self.labels_test == 0]
        if len(normals) == 0: # pragma: no cover
            self.skipTest("Não há não fraudes suficientes para o teste")

        test_set = [row.tolist() for row in normals]
        test_labels =  [0]*len(normals)
        self._run_test_set(test_set, test_labels)


    def test_model_ninety_fraud(self):
        """90% não fraudes e 10% fraudes"""
        frauds = self.ds_test[self.labels_test == 1]
        normals = self.ds_test[self.labels_test == 0]

        #O número de fraudes é menor que o número de transações normais
        #então limitamos o tamanho total do teste para manter
        #a proporção de 9:1
        total_test = min(len(frauds), len(normals))
        if total_test == 0: # pragma: no cover
            self.skipTest("Não há dados suficientes para o teste 90/10")

        n_fraud = max(1, int(total_test * 0.1)) # 10%
        n_normal = int(n_fraud * 9) # 90%

        test_set = [row.tolist() for row in frauds[:n_fraud]]
        test_set += [row.tolist() for row in normals[:n_normal]]
        test_labels = [1] * n_fraud + [0] * n_normal
        self._run_test_set(test_set, test_labels)


    def test_model_half_fraud(self):
        """50% fraudes e 50% não fraudes"""
        frauds = self.ds_test[self.labels_test == 1]
        normals = self.ds_test[self.labels_test == 0]

        #O número de fraudes é menor que o número de transações normais
        #então limitamos o tamanho total do teste para manter
        #a proporção de 5:5
        total_test = min(len(frauds), len(normals))
        if total_test == 0: # pragma: no cover
            self.skipTest("Não há dados suficientes para o teste 50/50")

        test_set = [row.tolist() for row in frauds[:total_test]]
        test_set += [row.tolist() for row in normals[:total_test]]
        test_labels = [1] * total_test + [0] * total_test
        self._run_test_set(test_set, test_labels)


    # ---------- Cobertura ----------
    def test_train_runs(self):
        """Verifica se train() roda sem erros"""
        try:
            history = self.model.train(epochs=1)
        except Exception as e: # pragma: no cover
            self.fail(f"train() levantou uma exceção: {e}")
        self.assertTrue(len(history.history['loss']) >= 1,
                        "train() não retornou histórico válido")

    def test_evaluate_runs(self):
        """Verifica se evaluate roda sem erros"""
        try:
            self.model.evaluate()
        except Exception as e: # pragma: no cover
            self.fail(f"evaluate() levantou uma exceção: {e}")


    def test_save_and_load_model(self):
        """Testa salvar e carregar o modelo"""
        temp_file = "temp_autoencoder.keras"

        self.model.save(temp_file)

        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "saved", temp_file)
        model_path = os.path.abspath(model_path)
        self.assertTrue(os.path.exists(model_path))

        model2 = AutoencoderFraudDetector()
        model2.load(temp_file) #Carrega modelo

        fraud_row = self.ds_test[self.labels_test == 1][0].tolist()
        _ = model2.predict(fraud_row)

        os.remove(model_path)  #Remove arquivo temp

    def test_save_runtime_raises(self):
        """Verifica se save levanta RuntimeError quando não há modelo"""
        model = AutoencoderFraudDetector()
        model.autoencoder = None # o autoencoder é definido no
        # __init__ então nunca é None,
        # mas para testar a função é necessário definir como None.
        try:
            model.save("teste.keras")
        except RuntimeError as e:
            self.assertIn("Nenhum modelo treinado", str(e))

    def test_load_filenotfound_raises(self):
        """Verifica se load levanta FileNotFoundError quando o arquivo não existe"""
        model = AutoencoderFraudDetector()
        try:
            model.load("arquivo_inexistente.keras")
        except FileNotFoundError as e:
            self.assertIn("não encontrado", str(e))

    def test_split_train_test(self):
        """Verifica se split_train_test divide corretamente o dataset"""
        ds_train, ds_val, ds_test, labels_test = split_train_test()
        # Verifica se retornou 4 elementos
        self.assertEqual(len([ds_train, ds_val, ds_test, labels_test]), 4)
        # Verifica se são arrays numpy
        self.assertIsInstance(ds_train, np.ndarray)
        self.assertIsInstance(ds_val, np.ndarray)
        self.assertIsInstance(ds_test, np.ndarray)
        self.assertIsInstance(labels_test, np.ndarray)

    def test_returns_dataframe(self):
        """Verifica se load_dataset retorna um DataFrame"""
        df = load_dataset()
        self.assertIsInstance(df, pd.DataFrame)

    def test_amount_standardized(self):
        """Verifica se a coluna Amount foi padronizada pelo StandardScaler (média≈0, desvio≈1)"""
        df = load_dataset()

        mean = df["Amount"].mean()
        std = df["Amount"].std()

        tol = 0.1
        self.assertTrue(abs(mean - 0.0) < tol,
                        f"Média de Amount não está próxima de 0 (valor: {mean})")
        self.assertTrue(abs(std - 1.0) < tol,
                        f"Desvio padrão de Amount não está próximo de 1 (valor: {std})")


    def test_evaluate_raises_without_threshold(self):
        """Deve lançar RuntimeError se threshold for None"""
        model = AutoencoderFraudDetector()  # cria modelo separado
        model.threshold = None  # garante o estado
        try:
            model.evaluate()
        except RuntimeError as e:
            self.assertIn("Threshold não definido", str(e))


    def test_predict_raises_without_threshold(self):
        """Deve lançar RuntimeError se threshold for None"""
        model = AutoencoderFraudDetector()  # cria modelo separado
        model.threshold = None  # garante o estado
        try:
            model.predict([0.1, 0.2, 0.3])
        except RuntimeError as e:
            self.assertIn("Threshold não definido", str(e))
