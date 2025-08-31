import os
import json

def load_environment():
    with open("utils/kaggle.json", "r") as f:
        creds = json.load(f)

    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

def load_dataset():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dataset = "mlg-ulb/creditcardfraud"
    api.dataset_download_files(dataset, path="data/", unzip=True)

if __name__ == "__main__":
    load_environment()

    load_dataset()
    
    print("Download concluído!")