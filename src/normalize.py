from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def min_max():
    print("Loading csv...")
    df = pd.read_csv('data/creditcard.csv')

    scaler = MinMaxScaler()

    df_to_scale = df[['Amount', 'Time']]
    df_scaled = scaler.fit_transform(df_to_scale)

    df[['Amount', 'Time']] = df_scaled

    return df 


if __name__ == "__main__":

    min_max()

    print("Normalização de Amount e Time realizada com sucesso!") 
