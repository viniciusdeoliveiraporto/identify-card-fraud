"""Esse módulo serve para extração do dataset
"""
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore

def load_environment():
    """Essa função carrega o ambiente do kaggle
    """
    try:
        with open("utils/kaggle.json", "r", encoding="uft-8") as f:
            creds = json.load(f)

        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
    except FileNotFoundError:
        print("Arquivo kaggle.json não encontrado em 'utils/'. Verifique o caminho.")
        raise
    except json.JSONDecodeError:
        print("Erro ao ler kaggle.json. Verifique se o arquivo está em formato JSON válido.")
        raise
    except KeyError as e:
        print(f"Chave faltando no kaggle.json: {e}")
        raise

def load_dataset():
    """Essa função carrega o dataset
    """
    api = KaggleApi()
    api.authenticate()

    dataset = "mlg-ulb/creditcardfraud"
    api.dataset_download_files(dataset, path="data/", unzip=True)

if __name__ == "__main__":
    load_environment()
    load_dataset()
    print("Download concluído!")
