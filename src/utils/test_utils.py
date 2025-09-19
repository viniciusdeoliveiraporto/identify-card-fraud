"""Esse módulo contém funções úteis para testes
"""
import csv
import math
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_id.txt")

def get_data(count: int) -> dict:
    "Busca os dados para os testes do csv, sendo count a quantidade" 
    " que não é fraude e 0,172% de count que é fraude"

    data = {"fraud": [], "legit": []}
    fraud_count = math.ceil(count * 0.172)

    with open(OUTPUT_PATH, "w", encoding="utf-8"):
        pass

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        i = 0
        for row in reader:
            if row[-1] == "1" and len(data["fraud"]) < fraud_count:
                data["fraud"].append(row)
                write_data(i, 1)
            elif row[-1] == "0" and len(data["legit"]) < count:
                data["legit"].append(row)
                write_data(i, 0)

            if len(data["legit"]) >= count and len(data["fraud"]) >= fraud_count:
                break
            i += 1

    return data

def write_data(index: int, type_data: int):
    """Essa função escreve dados
    """
    with open(OUTPUT_PATH, "a", encoding="utf-8") as file:
        file.write(f"{index}, {type_data}\n")
