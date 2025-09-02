import csv
import math
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_id.txt")

def get_data(count: int) -> dict:
    """Busca os dados para os testes do csv, sendo count a quantidade que não é fraude e 0,172% de count que é fraude"""

    data = {"fraud": [], "legit": []}
    fraud_count = math.ceil(count * 0.172)

    with open(OUTPUT_PATH, "w") as f:
        pass 

    with open(DATA_PATH, "r") as file:
        reader = csv.reader(file)
        header = next(reader) 
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

def write_data(id: int, type: int):
    with open(OUTPUT_PATH, "a") as file:
        file.write(f"{id}, {type}\n")

