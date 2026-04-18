import numpy as np
from pathlib import Path
import joblib
import json
import pandas as pd

def save_file(path: str, mappings: dict):
  with open(path, "w") as f:
    json.dump(mappings, f)

def load_file(path: str):
  with open(path, "r") as f:
    return json.load(f)

def load_model(path: str):
  path = Path(path)
  return joblib.load(path)

def save_model(model, path: str):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(model, path)

def haversine(lon1, lat1, lon2, lat2):
  """
    Converts two points in latitude and longitude to distance in kilometers.
  """
  lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  c = 2 * np.arcsin(np.sqrt(a))
  R = 6371  # Earth radius in km
  return R * c

def load_dataset(path, target_column):
  """
    Loads a dataset from a given path. Separates the target column from the features.
  """
  df = pd.read_csv(path)
  X = df.drop(columns=[target_column])
  y = df[target_column]
  return X, y