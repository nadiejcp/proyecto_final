"""
Script para descargar y extraer los datos originales del proyecto.
"""

import os
import urllib.request
import tarfile
from pathlib import Path
from utils.functions import haversine
import pandas as pd

class DataLoader:

    def __init__(self, config: dict):
        self.config = config

    def fetch_data(self):
        """
        INSTRUCCIONES:
        1. Asegúrate de que el directorio `data_path` exista (usa os.makedirs o Path.mkdir).
        2. Usa urllib.request.urlretrieve para descargar el archivo .tgz desde `data_url`.
        3. Usa tarfile.open para extraer el contenido en `data_path`.

        URL de los datos: "https://github.com/ageron/data/raw/main/housing.tgz"
        Ruta de destino recomendada: "data/raw/"
        """
        data_path = self.config["data"]["path_to_raw"]
        data_url = self.config["data"]["path_to_download"]
        data_dir = Path(data_path)
        if len(os.listdir(data_dir)) <= 1:
            data_dir.mkdir(parents=True, exist_ok=True)
            file_path = data_dir / "file.tar.gz"
            urllib.request.urlretrieve(data_url, file_path)
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(data_dir)
        print(f"Data downloaded and extracted to {data_dir}")

    def transform(self):
        """
        INSTRUCCIONES:
        1. Crea las siguientes columnas:
            - rooms_per_household: total_rooms / households
            - bedrooms_per_room: total_bedrooms / total_rooms
            - persons_per_house: population / households
            - income_per_person: median_income / population
            - rooms_per_person: total_rooms / population
            - bedrooms_per_household: total_bedrooms / households
        2. Calcula la distancia a la ciudad de San Francisco (playa) usando la función haversine.
        3. Devuelve el DataFrame transformado.
        """
        if not os.path.exists(self.config["data"]["path_to_processed"] + "/processed_data.csv") or self.config["data"]["reload"]:
            df = pd.read_csv(self.config["data"]["path_to_raw"] + "/housing/housing.csv")
            df = df.dropna()
            df['rooms_per_household'] = df['total_rooms'] / df['households']
            df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
            df['persons_per_house'] = df['population'] / df['households']
            df['income_per_person'] = df['median_income'] / df['population']
            df['rooms_per_person'] = df['total_rooms'] / df['population']
            df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
            for col in self.config["features"]["one_hot_encode"]:
                df[col] = pd.factorize(df[col])[0] + 1

            cols_to_drop = self.config["features"].get("to_drop", None)
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            df.to_csv(self.config["data"]["path_to_processed"] + "/processed_data.csv", index=False)
        print(f"Data transformed and saved to {self.config['data']['path_to_processed']}")


