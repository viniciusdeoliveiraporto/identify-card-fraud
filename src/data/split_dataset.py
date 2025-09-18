import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.dataset_utils import *

def split_train_test():
    print("Dividindo dataset em treino e teste...")
    df = load_dataset()

    df_normal = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]

    df_train, df_val = train_test_split(df_normal, test_size=0.2, random_state=42)

    df_test = pd.concat([df_val, df_fraud])

    ds_train = df_train.drop(columns=['Class']).values
    ds_val = df_val.drop(columns=['Class']).values
    ds_test = df_test.drop(columns=['Class']).values
    labels_test = df_test['Class'].values

    return ds_train, ds_val, ds_test, labels_test

def split_train_test_sample():
    print("Dividindo dataset em treino e teste...")
    df = load_dataset_test()

    df_normal = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]
    df_train, df_val = train_test_split(df_normal, test_size=0.2, random_state=42)
    df_test = pd.concat([df_val, df_fraud])

    ds_train = df_train.drop(columns=['Class']).values
    ds_val = df_val.drop(columns=['Class']).values
    ds_test = df_test.drop(columns=['Class']).values
    labels_test = df_test['Class'].values

    return ds_train, ds_val, ds_test, labels_test
