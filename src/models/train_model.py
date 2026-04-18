"""
Script de entrenamiento para el modelo final y evaluación básica.
"""

import pandas as pd
import joblib
from pathlib import Path
from utils.functions import save_model, load_model, load_dataset
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import os
from math import sqrt

# IMPORTANTE: Se debe importar los algoritmos que quieran usar, por ejemplo:
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

class Trainer:
    def __init__(self, config: dict, model_factory):
        self.config = config
        self.model_factory = model_factory
    
    def train_best_model(self, model_name: str):
        """
        INSTRUCCIONES:
        1. Carga los datos de entrenamiento procesados (que ya pasaron por `build_features.py`).
        2. Separa las características (X) de la etiqueta a predecir (y = 'median_house_value').
        3. Instancia tu mejor modelo encontrado después de la fase de experimentación y "fine tuning"
        (Por ejemplo: RandomForestRegressor con los mejores hiperparámetros).
        4. Entrena el modelo haciendo fit(X, y).
        5. Guarda el modelo entrenado en `model_save_path` (ej. 'models/best_model.pkl') usando joblib.dump().
        """
        if os.path.exists(self.config['models'][model_name]['save_path']) and not self.config['data']['reload']:
            return load_model(self.config['models'][model_name]['save_path'])
        X_train, y_train = load_dataset(self.config["data"]["interim_data_path"] + "/train_set.csv", self.config["features"]["target"])
        model = self.model_factory.create(model_name)
        model.fit(X_train, y_train)
        save_model(model, self.config['models'][model_name]['save_path'])

    def evaluate_model(self, model_name: str, processed_test_data_path: str):
        """
        INSTRUCCIONES:
        1. Carga el modelo guardado con joblib.load().
        2. Carga los datos de prueba preprocesados.
        3. Genera predicciones (y_pred) sobre los datos de prueba usando predict().
        4. Compara y_pred con las etiquetas reales calculando el RMSE y repórtalo en la terminal.
        """
        model = load_model(self.config['models'][model_name]['save_path'])
        X_eval, y_eval = load_dataset(self.config["data"]["interim_data_path"] + processed_test_data_path, self.config["features"]["target"])
        y_pred = model.predict(X_eval)
        mae = mean_absolute_error(y_eval, y_pred)
        rmse = sqrt(mean_squared_error(y_eval, y_pred))
        r2 = r2_score(y_eval, y_pred)
        print(model_name, f"MAE: {mae}")
        print(model_name, f"RMSE: {rmse}")
        print(model_name, f"R2: {r2}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

