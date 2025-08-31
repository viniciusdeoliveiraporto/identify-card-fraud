from sklearn.model_selection import train_test_split
import pandas as pd
import utils.dataset_utils as normalize

def split_train_test():
    print("Dividindo dataset em treino e teste...")
    df_verdadeiro = normalize.min_max()

    df = df_verdadeiro.head(600)

    df_normal = df[df["Class"] == 0]
    df_fraud = df[df["Class"] == 1]

    df_train, df_val = train_test_split(df_normal, test_size=0.2, random_state=40)

    df_test = pd.concat([df_val, df_fraud])
    labels_test = pd.concat([df["Class"][df_val.index], df["Class"][df_fraud.index]])

    return df_train, df_test, labels_test


if __name__ == "__main__":
    split_train_test()
    print("Dataset dividido em treino e teste")