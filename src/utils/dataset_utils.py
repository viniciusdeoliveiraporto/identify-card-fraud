"""Esse módulo possui funções úteis para manipular o dataset 
"""
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_dataset():
    """Essa função carrega o dataset 
    """
    print("Carregando csv...")
    base_path = os.path.dirname(__file__)
    data_folder = os.path.join(base_path, "../data")  # ../data porque estamos em utils

    # Caminhos dos arquivos
    real_database_path = os.path.join(data_folder, "creditcard.csv")
    sample_database_path = os.path.join(data_folder, "sample.csv")

    # Tentativa de carregar o CSV
    try:
        df = pd.read_csv(real_database_path)
        print("creditcard.csv carregado com sucesso!")
    except FileNotFoundError:
        print("creditcard.csv não encontrado, tentando sample.csv...")
        try:
            df = pd.read_csv(sample_database_path)
            print("sample.csv carregado com sucesso!")
        except FileNotFoundError:
            raise FileNotFoundError("Nenhum CSV encontrado! Verifique o" +
                                        "caminho para creditcard.csv ou sample.csv.")

    # Remover a coluna Time e normalizar Amount
    # scaler_amount = MinMaxScaler()
    df = df.drop(columns=["Time"])
    scaler_amount = StandardScaler()
    df[["Amount"]] = scaler_amount.fit_transform(df[["Amount"]])

    # Escalonar V1–V28
    pca_cols = [f"V{i}" for i in range(1, 29)]
    scaler_pca = StandardScaler()
    df[pca_cols] = scaler_pca.fit_transform(df[pca_cols])

    return df

if __name__ == "__main__":
    load_dataset()
    print("Normalização dos dados realizada com sucesso!")
