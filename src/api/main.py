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
VALID_OCEAN_PROXIMITY = {"<1H OCEAN", "NEAR OCEAN", "INLAND"}

class HousingFeatures(BaseModel):
    longitude:          float
    latitude:           float
    housing_median_age: float
    total_rooms:        float
    total_bedrooms:     float
    population:         float
    households:         float
    median_income:      float
    ocean_proximity:    str   # "<1H OCEAN" | "NEAR OCEAN" | "INLAND"

    # --- Validadores de campos numéricos ---
    @field_validator(
        "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "median_income",
        mode="before",
    )
    @classmethod
    def must_be_finite_number(cls, v, info):
        try:
            num = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"'{info.field_name}' debe ser un número.")
        if math.isnan(num) or math.isinf(num):
            raise ValueError(f"'{info.field_name}' no puede ser NaN ni infinito.")
        return num

    # --- Validador de ocean_proximity ---
    @field_validator("ocean_proximity", mode="before")
    @classmethod
    def must_be_valid_proximity(cls, v):
        if str(v).strip().upper() not in {x.upper() for x in VALID_OCEAN_PROXIMITY}:
            raise ValueError(
                f"ocean_proximity inválido. Opciones: {sorted(VALID_OCEAN_PROXIMITY)}"
            )
        return str(v).strip().upper()

# ---------------------------------------------------------------------------
# Modelo global
# ---------------------------------------------------------------------------
model = None

@app.on_event("startup")
def load_model():
    """Carga el modelo al iniciar el servidor."""
    global model
    model_path = os.path.join(BASE_DIR, "..", "..", "models", "best_model.pkl")
    try:
        model = joblib.load(model_path)
        print(f"[OK] Modelo cargado desde: {model_path}")
    except Exception as e:
        print(f"[WARN] No se pudo cargar el modelo: {e}")
        print("      Ya lo entrenaste y guardaste en models/best_model.pkl?")

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

    # Construir DataFrame de entrada
    data = pd.DataFrame([{
        "longitude":          features.longitude,
        "latitude":           features.latitude,
        "housing_median_age": features.housing_median_age,
        "total_rooms":        features.total_rooms,
        "total_bedrooms":     features.total_bedrooms,
        "population":         features.population,
        "households":         features.households,
        "median_income":      features.median_income,
        "ocean_proximity":    features.ocean_proximity,
    }])

    try:
        prediction = float(model.predict(data)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    return {"predicted_price": prediction}


# ---------------------------------------------------------------------------
# Cómo correr la API:
#   uvicorn src.api.main:app --reload
# ---------------------------------------------------------------------------
