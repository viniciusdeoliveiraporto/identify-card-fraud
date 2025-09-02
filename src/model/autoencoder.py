import tensorflow
import numpy as np
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import classification_report, confusion_matrix
from src.data.split_dataset import split_train_test

class AutoencoderFraudDetector:
    def __init__(self):
        self.ds_train, self.ds_test, self.labels_test = split_train_test()
        input_dim = self.ds_train.shape[1] 

        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(16, activation="relu")(input_layer)
        encoder = Dense(8, activation="relu")(encoder)

        # Decoder
        decoder = Dense(16, activation="relu")(encoder)
        decoder = Dense(input_dim, activation="linear")(decoder)

        # Modelo
        self.autoencoder = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder.compile(optimizer="adam", loss="mse")

    # ----------------- NOVO: treinar, avaliar e adivinhar -----------------
    def train(self, epochs=25, batch_size=32):
        history = self.autoencoder.fit(
            self.ds_train, self.ds_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True
        )
        return history

    def evaluate(self, threshold_percentile=95):
        reconstructions = self.autoencoder.predict(self.ds_test)
        mse = np.mean(np.power(self.ds_test - reconstructions, 2), axis=1)

        threshold = np.percentile(mse, threshold_percentile)
        y_pred = (mse > threshold).astype(int)

        print(confusion_matrix(self.labels_test, y_pred))
        print(classification_report(self.labels_test, y_pred))

    def predict(self, row, threshold_percentile=95):
        row_array = np.array([float(x) for x in row]).reshape(1, -1)

        reconstruction = self.autoencoder.predict(row_array)
        mse = np.mean(np.power(row_array - reconstruction, 2))

        reconstructions_test = self.autoencoder.predict(self.ds_test)
        mse_test = np.mean(np.power(self.ds_test - reconstructions_test, 2), axis=1)
        threshold = np.percentile(mse_test, threshold_percentile)

        return int(mse > threshold)

    # ----------------- NOVO: salvar e carregar -----------------
    def save(self, filename="autoencoder.keras"):
        """Salva apenas o modelo Keras dentro de src/model/saved"""
        if self.autoencoder is None:
            raise RuntimeError("Nenhum modelo treinado para salvar.")
        
        save_dir = os.path.join(os.path.dirname(__file__), "saved")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, filename)

        self.autoencoder.save(model_path)
        print(f"Modelo salvo em: {model_path}")

    def load(self, filename="autoencoder.keras"):
        """Carrega apenas o modelo Keras de src/model/saved"""
        model_path = os.path.join(os.path.dirname(__file__), "saved", filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo {model_path} não encontrado")
        
        self.autoencoder = tensorflow.keras.models.load_model(model_path)
        print(f"Modelo carregado de: {model_path}")
        return self