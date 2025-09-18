import os
import tensorflow
import json
import numpy as np

# Silencia avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dropout, Input, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
from src.data.split_dataset import split_train_test

class AutoencoderFraudDetector:
    def _init_(self):
        self.ds_train, self.ds_val, self.ds_test, self.labels_test = split_train_test()
        self.input_dim = self.ds_train.shape[1] 
        self.threshold = None
        
        input_layer = Input(shape=(self.input_dim,))

        # ---------------- Encoder ----------------
        encoded_32 = Dense(32, activation="relu")(input_layer)
        encoded_32 = Dropout(0.1)(encoded_32)

        encoded_14 = Dense(14, activation="relu")(encoded_32)
        encoded_14 = Dropout(0.1)(encoded_14)

        encoded_7 = Dense(7, activation="relu")(encoded_14)
        encoded_7 = Dropout(0.1)(encoded_7)

        # ---------------- Decoder ----------------
        decoded_7 = Dense(7, activation="relu")(encoded_7)
        decoded_7 = Dropout(0.1)(decoded_7)

        decoded_32 = Dense(32, activation="relu")(decoded_7)
        decoded_32 = Dropout(0.1)(decoded_32)

        output_layer = Dense(self.input_dim, activation="linear")(decoded_32)

        # ---------------- Modelo ----------------
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.autoencoder.compile(optimizer="adam", loss="mse")

    # ----------------- Treinar modelo, Avaliar e Adivinhar transações -----------------
    def train(self, epochs=10, batch_size=128, threshold_percentile=96):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            min_delta=1e-4,
            restore_best_weights=True
        )

        history = self.autoencoder.fit(
            self.ds_train, self.ds_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data =(self.ds_val, self.ds_val),
            shuffle=True,
            callbacks=[early_stop]
        )
        
        #if self.threshold is None:
        reconstructions_val = self.autoencoder.predict(self.ds_val)
        mse_val = np.mean(np.power(self.ds_val - reconstructions_val, 2), axis=1)
        self.threshold = np.percentile(mse_val, threshold_percentile)
        print(f"\nThreshold fixo calculado: {self.threshold:.6f}\n")

        return history

    def evaluate(self):
        if self.threshold is None:
            raise RuntimeError("Threshold não definido. Treine o modelo antes de avaliar.")

        reconstructions_test = self.autoencoder.predict(self.ds_test)
        mse_test = np.mean(np.power(self.ds_test - reconstructions_test, 2), axis=1)
        y_pred = (mse_test > self.threshold).astype(int)

        print(confusion_matrix(self.labels_test, y_pred))
        print(classification_report(self.labels_test, y_pred))

    def predict(self, row):
        if self.threshold is None:
            raise RuntimeError("Threshold não definido. Treine o modelo antes de avaliar.")
        
        row_array = np.array([float(x) for x in row]).reshape(1, -1)

        reconstruction = self.autoencoder.predict(row_array)
        mse_row = np.mean(np.power(row_array - reconstruction, 2))

        return int(mse_row > self.threshold)

    # ----------------- Salvar e Carregar modelo -----------------
    def save(self, filename="autoencoder.keras"):
        if self.autoencoder is None:
            raise RuntimeError("Nenhum modelo treinado para salvar.")

        save_dir = os.path.join(os.path.dirname(__file__), "saved")
        os.makedirs(save_dir, exist_ok=True)

        # Salva o modelo keras
        model_path = os.path.join(save_dir, filename)
        self.autoencoder.save(model_path)

        # Salva os metadados extras
        metadata = {
            "threshold": self.threshold
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        print(f"Modelo e metadados salvos em: {save_dir}")


    def load(self, filename="autoencoder.keras"):
        model_path = os.path.join(os.path.dirname(__file__), "saved", filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo {model_path} não encontrado")

        self.autoencoder = tensorflow.keras.models.load_model(model_path)

        # Carrega os metadados extras
        metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.threshold = metadata.get("threshold", None)

        print(f"Modelo carregado de: {model_path}")
        return self