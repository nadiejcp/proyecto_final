from sklearn.linear_model import LinearRegression
from .base import BaseModel

class BaselineModel(BaseModel):
  def __init__(self, **kwargs):
    self.model = LinearRegression()

  def fit(self, X, y):
    self.model.fit(X, y)
    return self

  def predict(self, X):
    return self.model.predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)
