"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def split_and_save_data(config: dict):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    interim_data_path = config["data"]["interim_data_path"]
    if not os.path.exists(interim_data_path + "/train_set.csv") or config["data"]["reload"]:
        data_path = config["data"]["path_to_processed"] + "/processed_data.csv"
        test_size = config["data"]["split"]["test_size"]
        df = pd.read_csv(data_path)
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=42)
        train_set.to_csv(interim_data_path + "/train_set.csv", index=False)
        test_set.to_csv(interim_data_path + "/test_set.csv", index=False)
    print(f"Data split and saved to {interim_data_path}")

if __name__ == "__main__":
    # RAW_PATH = "data/raw/housing/housing.csv"
    # INTERIM_PATH = "data/interim/"
    # split_and_save_data(RAW_PATH, INTERIM_PATH)
    print("Script para dividir datos... (Falta el código!)")
