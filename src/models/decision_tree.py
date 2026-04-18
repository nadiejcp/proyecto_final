
from sklearn.tree import DecisionTreeRegressor
from .base import BaseModel

class DecisionTreeModel(BaseModel):
  def __init__(self, **kwargs):
    self.model = DecisionTreeRegressor()
    if 'use_params' in kwargs:
      del kwargs['use_params']
      self.model = DecisionTreeRegressor(**kwargs)

  def fit(self, X, y):
    self.model.fit(X, y)
    return self

  def predict(self, X):
    return self.model.predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)

