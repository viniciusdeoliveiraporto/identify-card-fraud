from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

def min_max():
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
            print("csample.csv carregado com sucesso!")
        except FileNotFoundError:
            raise FileNotFoundError("Nenhum CSV encontrado! Verifique o caminho para real_database.csv ou creditcard.csv.")

    scaler = MinMaxScaler()
    df_to_scale = df[['Amount', 'Time']]
    df_scaled = scaler.fit_transform(df_to_scale)

    df[['Amount', 'Time']] = df_scaled

    return df 

if __name__ == "__main__":
    min_max()
    print("Normalização de Amount e Time realizada com sucesso!") 
