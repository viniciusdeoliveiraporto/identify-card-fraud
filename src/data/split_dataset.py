import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.dataset_utils import min_max 

def split_train_test():
    print("Dividindo dataset em treino e teste...")
    df_verdadeiro = min_max()

    df = df_verdadeiro.head(600)

    df_normal = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]

    df_train, df_val = train_test_split(df_normal, test_size=0.2, random_state=40)

    df_test = pd.concat([df_val, df_fraud])

    ds_train = df_train.drop(columns=['Class']).values  
    ds_test = df_test.drop(columns=['Class']).values  
    labels_test = df_test['Class'].values  

    return ds_train, ds_test, labels_test


if __name__ == "__main__":
    split_train_test()
    print("Dataset dividido em treino e teste")