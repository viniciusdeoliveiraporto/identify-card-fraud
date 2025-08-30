import numpy as np
import pandas as pd
from typing import List, Union

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

import os
import joblib

class FraudDetectionModel:
    def __init__(self, model_dir: str = "saved_model"):
        self.model_dir = model_dir
        # Garante que o diretório exista
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Artefatos que serão criados em train() ou load() 
        self.autoencoder = None 
        self.scaler_time = None 
        self.scaler_pca = None 
        self.scaler_amount = None 
        self.input_columns = None # ordem das colunas (sem 'Class') 
        self.threshold = None

    def _build_autoencoder(self):
        pass

    def predict(self, data: List[Union[str, float, int]]) -> int:
        """
        Recebe uma linha (lista) no mesmo formato do CSV (pode incluir label no final).
        Retorna 1 se for considerado fraude, 0 caso contrário.
        """
        # 1) Carregar modelo se necessário
        if self.autoencoder is None or self.scaler_amount is None or self.scaler_time is None:
            try:
                self.load()
            except Exception:
                raise RuntimeError("Modelo não treinado nem encontrado em disco. Chame train() primeiro ou coloque o modelo em saved_model/")

        # 2) Converter e Pré-processar
        try:
            x_scaled = self._preprocess_row(data)  # shape (1, D)
        except Exception as e:
            raise ValueError(f"Erro ao preprocessar a linha: {e}")

        # 3) Reconstruir e calcular mse
        recon = self.autoencoder.predict(x_scaled)
        mse = float(np.mean(np.power(x_scaled - recon, 2)))
        return 1 if mse > self.threshold else 0
    

    def train(self, 
              csv_path: str ="creditcard.csv", 
              epochs: int = 10, 
              batch_size: int = 256, 
              validation_split: float = 0.2, 
              verbose: int =1, 
              sample_legit: int = None, 
              threshold_percentile: float = 99.0):
        """
        Treina o Autoencoder apenas com transações legítimas (Class == 0).
        
        Parâmetros:
        - csv_path: caminho para o dataset creditcard.csv.
        - epochs: número de épocas (quantas vezes o modelo vê o dataset completo).
        - batch_size: tamanho dos lotes de dados processados por vez.
        - validation_split: proporção dos dados separados para validação.
        - verbose: nível de verbosidade do treinamento (0 = silencioso, 1 = normal, 2 = detalhado).
        - sample_legit: se definido, limita a quantidade de dados usados no treino
        - threshold_percentile: percentil usado para definir o limiar de reconstrução.
        """

        # 1) Carregar o dataset inteiro e verificar se contém a coluna "Class"
        dataframe = pd.read_csv(csv_path)
        if "Class" not in dataframe.columns:
            raise ValueError("CSV deve conter coluna 'Class'")

        # 2) Selecionar apenas as transações legítimas (Class == 0)
        legitimo_df = dataframe[dataframe["Class"] == 0].copy()

        # 3) Se sample_legit for definido, pegar apenas uma amostra do conjunto de dados
        if sample_legit is not None:
            legitimo_df = legitimo_df.sample(n=min(sample_legit, len(legitimo_df)), random_state=40)

        # 4) Remove a coluna "Class" e converter os valores para float
        variaveis = legitimo_df.drop(columns=["Class"])

        # 5) Separar em treino e validação
        treino_var, validacao_var = train_test_split(variaveis, test_size=validation_split, random_state=40)

        # 6) Escalar variaveis
        treino_var_escalado, validacao_var_escalado = self._fit_scalers_and_transform(treino_var, validacao_var)

        # 7) Construir o Autoencoder com a dimensão correta de entrada
        dimensao_entrada = treino_var_escalado.shape[1]
        self.autoencoder = self._build_autoencoder(dimensao_entrada)

        # 8) Parar treinamento cedo para evitar overfitting
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]

        # 8) Executar o treinamento do Autoencoder (entrada == saída)
        self.autoencoder.fit(
            treino_var, validacao_var,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(treino_var_escalado, validacao_var_escalado),
            verbose=verbose,
            callbacks=callbacks
        )

        # 9) Calcular erro de reconstrução no conjunto de validação e definir o limiar
        reconstrucoes = self.autoencoder.predict(validacao_var_escalado)
        erros_reconstrucao = np.mean(np.power(validacao_var_escalado - reconstrucoes, 2), axis=1)
        self.limiar_reconstrucao = float(np.percentile(erros_reconstrucao, threshold_percentile))

        # 10) Salvar artefatos (modelo, scaler e threshold) para uso futuro
        self.save()

        # 11) Retornar os valores do limiar e do erro médio de validação
        return {
            "limiar_reconstrucao": self.limiar_reconstrucao,
            "media_mse_validacao": float(erros_reconstrucao.mean())
        }

    def _preprocess_row(self):
        pass

    def _fit_scalers_and_transform(self):
        pass

    def save(self):
        """Salva o modelo e artefatos em disco"""
        if self.autoencoder is None:
            raise RuntimeError("Nenhum modelo treinado para salvar.")

        # Salva modelo Keras
        model_path = os.path.join(self.model_dir, "autoencoder.keras")
        self.autoencoder.save(model_path)

        # Salva scalers, colunas e threshold
        artifacts = {
            "scaler_time": self.scaler_time,
            "scaler_pca": self.scaler_pca,
            "scaler_amount": self.scaler_amount,
            "input_columns": self.input_columns,
            "threshold": self.threshold,
        }
        artifacts_path = os.path.join(self.model_dir, "artifacts.pkl")
        joblib.dump(artifacts, artifacts_path)

    def load(self):
        """Carrega modelo e artefatos do disco"""
        model_path = os.path.join(self.model_dir, "autoencoder.keras")
        artifacts_path = os.path.join(self.model_dir, "artifacts.pkl")

        if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
            raise FileNotFoundError("Modelo ou artefatos não encontrados em disco.")

        # Carrega modelo Keras
        self.autoencoder = tf.keras.models.load_model(model_path)

        # Carrega scalers, colunas e threshold
        artifacts = joblib.load(artifacts_path)
        self.scaler_time = artifacts.get("scaler_time")
        self.scaler_pca = artifacts.get("scaler_pca")
        self.scaler_amount = artifacts.get("scaler_amount")
        self.input_columns = artifacts.get("input_columns")
        self.threshold = artifacts.get("threshold")

        return self
    