from model.autoencoder import AutoencoderFraudDetector

if __name__ == "__main__":
    model = AutoencoderFraudDetector()
    
    print("Treinando modelo...")
    model.train(epochs=25)

    print("Avaliando modelo...")
    model.evaluate(threshold_percentile=95)

    print("Salvando modelo...")
    model.save("autoencoder.keras")

    print("Carregando modelo salvo...")
    loaded_model = AutoencoderFraudDetector().load("autoencoder.keras")

    print("Avaliando modelo carregado...")
    loaded_model.evaluate(threshold_percentile=95)