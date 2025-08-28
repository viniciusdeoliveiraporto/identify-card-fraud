import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import os
import pickle

class Model:
    def __init__(self, model_dir="saved_model"):
        self.model_dir = model_dir

    def predict(self, data: list[float]) -> int:
        return 1
    
    def train(self, csv_path="creditcard.csv", epochs=10, batch_size=256, validation_split=0.2, verbose=1, sample_legit=None, threshold_percentile=98):
        
        # 1) Carregar o dataset inteiro e verificar se contém a coluna "Class"
        dataframe = pd.read_csv(csv_path)
        if "Class" not in dataframe.columns:
            raise ValueError("CSV deve conter coluna 'Class'")

        # 2) Selecionar apenas as transações legítimas (Class == 0)
        legitimo_df = dataframe[dataframe["Class"] == 0].copy()
        # fraudes_df = dataframe[dataframe["Class"] == 1].copy()  # Não usado no momento

        # 3) Se sample_legit for definido, pegar apenas uma amostra do conjunto de dados
        if sample_legit is not None:
            legitimo_df = legitimo_df.sample(n=min(sample_legit, len(legitimo_df)), random_state=42)

        # 4) Remover a coluna "Class" e converter os valores para float
        variaveis = legitimo_df.drop(columns=["Class"]).values.astype(float)

        # 5) Normalizar/escalar as variáveis
        self.variaveis_scaler = StandardScaler()
        variaveis_escaladas = self.variaveis_scaler.fit_transform(variaveis)

        # 6) Separar em treino e validação
        treino_var, validacao_var = train_test_split(variaveis_escaladas, test_size=validation_split, random_state=42)

        # 7) Construir o Autoencoder com a dimensão correta de entrada
        dimensao_entrada = treino_var.shape[1]
        self.autoencoder = self._build_autoencoder(dimensao_entrada)

        # 8) Executar o treinamento do Autoencoder (entrada == saída)
        self.autoencoder.fit(
            treino_var, treino_var,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validacao_var, validacao_var),
            verbose=verbose
        )

        # 9) Calcular erro de reconstrução no conjunto de validação e definir o limiar
        reconstrucoes = self.autoencoder.predict(validacao_var)
        erros_reconstrucao = np.mean(np.power(validacao_var - reconstrucoes, 2), axis=1)
        self.limiar_reconstrucao = float(np.percentile(erros_reconstrucao, threshold_percentile))

        # 10) Salvar artefatos (modelo, scaler e threshold) para uso futuro
        self.save()

        # 11) Retornar os valores do limiar e do erro médio de validação
        return {
            "limiar_reconstrucao": self.limiar_reconstrucao,
            "media_mse_validacao": float(erros_reconstrucao.mean())
        }