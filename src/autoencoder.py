from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import split_dataset

ds_train, ds_test, labels_test = split_dataset.split_train_test()

input_dim = ds_train.shape[1]  # número de colunas

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(16, activation="relu")(input_layer)
encoder = Dense(8, activation="relu")(encoder)

# Decoder
decoder = Dense(16, activation="relu")(encoder)
decoder = Dense(input_dim, activation="linear")(decoder)

# Modelo Autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    ds_train, ds_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)

reconstructions = autoencoder.predict(ds_test)
mse = np.mean(np.power(ds_test - reconstructions, 2), axis=1)

threshold = np.percentile(mse, 110)  # 95º percentil
y_pred = (mse > threshold).astype(int)

print(confusion_matrix(labels_test, y_pred))
print(classification_report(labels_test, y_pred))