from src.model.autoencoder import AutoencoderFraudDetector

if __name__ == "__main__":
    print("Iniciando o sistema de detecção de fraudes...\n")

    start = True
    model = AutoencoderFraudDetector()

    print("\nOla! Esse é o prototipo inicial do sistema de detecção de anomalias/fraudes em transações de cartão de crédito.\n")

    while(start):
        user_input = input("Quais das opções abaixo você deseja realizar?\n" +
        "(1) Treinar autocodificador\n" +
        "(2) Avaliar autocodificador\n" +
        "(3) Salvar modelo\n" +
        "(4) Carregar modelo\n" +
        "(5) Sair\n")

        if user_input == "1":
            print("Treinando o autocodificador...\n")
            model.train(epochs=10, batch_size=128, threshold_percentile=96)
            print("Treinamento concluido!\n")
        elif user_input == "2":
            print("Avaliando o autocodificador...\n")
            model.evaluate()
            print("Avaliação concluida!\n")
        elif user_input == "3":
            print("Salvando o modelo...\n")
            model.save("autoencoder.keras")
            print("Modelo salvo!\n")
        elif user_input == "4":
            print("Carregando o modelo salvo...\n")
            model = AutoencoderFraudDetector().load("autoencoder.keras")
            print("Modelo carregado!\n")
        elif user_input == "5":
            print("Saindo do programa...\n")
            start = False
        else:
            print("Input invalido! Tente novamente.\n\n")

    