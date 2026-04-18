"""
API Básica usando FastAPI para servir el modelo entrenado.
Incluye interfaz gráfica servida desde /  y validación de datos en el backend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd
import math
import os
from config.load_config import load_config
from utils.functions import load_file

# ---------------------------------------------------------------------------
# Configuración de la app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)",
    version="1.0",
)

# Directorio base de este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Esquema de entrada con validación
# ---------------------------------------------------------------------------
VALID_OCEAN_PROXIMITY = {"<1H OCEAN", "NEAR OCEAN", "INLAND", "ISLAND", "NEAR BAY"}

class HousingFeatures(BaseModel):
    longitude:          float
    latitude:           float
    housing_median_age: float
    total_rooms:        float
    total_bedrooms:     float
    population:         float
    households:         float
    median_income:      float
    ocean_proximity:    str

# ---------------------------------------------------------------------------
# Modelo global
# ---------------------------------------------------------------------------
model = None
mappings = {}

@app.on_event("startup")
def load_model():
    """Carga el modelo al iniciar el servidor."""
    global model
    config = load_config("config/config.yaml")
    model_path = config['best_model']['save_path']
    global mappings
    mappings = load_file(config['data']['mappings'])
    mappings = mappings.get('ocean_proximity')
    try:
        model = joblib.load(model_path)
        print(f"[OK] Modelo cargado desde: {model_path}")
    except Exception as e:
        print(f"[WARN] No se pudo cargar el modelo: {e}")
        print("      Ya lo entrenaste y guardaste en models/best_model.joblib?")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    """Sirve la interfaz grafica."""
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Recibe las características de la vivienda y retorna el precio predicho.

    Pasos:
    1. Pydantic valida que todos los campos sean numéricos y ocean_proximity válido.
    2. Se construye un DataFrame con las mismas columnas usadas en entrenamiento.
    3. Se llama a model.predict() y se retorna el resultado.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no se ha cargado. Entrena y guarda el modelo primero.",
        )

    fields = ["longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "median_income"]
    
    for f in fields:
        try:
            v = getattr(features, f, None)
            num = float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{f} debe ser un numero: {e}")
        if math.isnan(num) or math.isinf(num):
            raise HTTPException(status_code=400, detail=f"'{f}' no puede ser NaN ni infinito.")
    global mappings
    category = mappings.get(str(features.ocean_proximity).strip().upper())
    if not category:
        raise HTTPException(
            status_code=400, detail=f"Proximidad al Oceano inválido. Opciones: {sorted(mappings)}"
        )
    df = pd.DataFrame([{
        "longitude":          features.longitude,
        "latitude":           features.latitude,
        "housing_median_age": features.housing_median_age,
        "total_rooms":        features.total_rooms,
        "total_bedrooms":     features.total_bedrooms,
        "population":         features.population,
        "households":         features.households,
        "median_income":      features.median_income,
        "ocean_proximity":    category,
    }])

    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['persons_per_house'] = df['population'] / df['households']
    df['income_per_person'] = df['median_income'] / df['population']
    df['rooms_per_person'] = df['total_rooms'] / df['population']
    df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']

    try:
        prediction = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    return {"predicted_price": prediction}


# ---------------------------------------------------------------------------
# Cómo correr la API:
#   uvicorn src.api.main:app --reload
# ---------------------------------------------------------------------------
