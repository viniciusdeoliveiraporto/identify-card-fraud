"""
Módulo principal para interagir com o sistema de detecção de fraudes.
"""

from src.model.autoencoder import AutoencoderFraudDetector


if __name__ == "__main__":
    print("Iniciando o sistema de detecção de fraudes...\n")

    START_FLAG = True
    model = AutoencoderFraudDetector()

    print(
        "\nOlá! Esse é o protótipo inicial do sistema de detecção de "
        "anomalias/fraudes em transações de cartão de crédito.\n"
    )

    while START_FLAG:
        user_input = input(
            "Quais das opções abaixo você deseja realizar?\n"
            "(1) Treinar autocodificador\n"
            "(2) Avaliar autocodificador\n"
            "(3) Salvar modelo\n"
            "(4) Carregar modelo\n"
            "(5) Reiniciar modelo\n"
            "(6) Sair\n"
        )

        if user_input == "1":
            print("Treinando o autocodificador...\n")
            model.train(epochs=10, batch_size=128, threshold_percentile=96)
            print("Treinamento concluído!\n")

        elif user_input == "2":
            print("Avaliando o autocodificador...\n")
            model.evaluate()
            print("Avaliação concluída!\n")

        elif user_input == "3":
            print("Salvando o modelo...\n")
            model.save("autoencoder.keras")
            print("Modelo salvo!\n")

        elif user_input == "4":
            print("Carregando o modelo salvo...\n")
            model = AutoencoderFraudDetector().load("autoencoder.keras")
            print("Modelo carregado!\n")

        elif user_input == "5":
            print("Reiniciando o modelo...\n")
            model = AutoencoderFraudDetector()
            print("Modelo reiniciado!\n")

        elif user_input == "6":
            print("Saindo do programa...\n")
            START_FLAG = False

        else:
            print("Entrada inválida! Tente novamente.\n")
