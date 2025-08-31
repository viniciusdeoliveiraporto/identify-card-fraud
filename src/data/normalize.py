from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

def min_max():
    print("Carregando csv...")
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, "creditcard.csv")
    df = pd.read_csv(csv_path)

    scaler = MinMaxScaler()
    df_to_scale = df[['Amount', 'Time']]
    df_scaled = scaler.fit_transform(df_to_scale)

    df[['Amount', 'Time']] = df_scaled

    return df 

if __name__ == "__main__":
    min_max()
    print("Normalização de Amount e Time realizada com sucesso!") 
