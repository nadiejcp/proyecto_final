from xgboost import XGBRegressor
from .base import BaseModel

class XGBoostModel(BaseModel):
  def __init__(self, **kwargs):
    self.model = XGBRegressor()
    if 'use_params' in kwargs:
      del kwargs['use_params']
      del kwargs['save_path']
      self.model = XGBRegressor(**kwargs)

  def fit(self, X, y):
    self.model.fit(X, y)
    return self

  def predict(self, X):
    return self.model.predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)