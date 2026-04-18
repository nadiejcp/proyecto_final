from sklearn.neural_network import MLPRegressor
from .base import BaseModel
from sklearn.preprocessing import StandardScaler

class MLPModel(BaseModel):
  def __init__(self, hidden_layers=None, **kwargs):
    """
    Initialize MLP Model.
    Args:
        hidden_layers (list): List of neuron counts for hidden layers.
        **kwargs: Additional arguments for MLPClassifier.
    """
    self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=1500)
    if 'use_params' in kwargs:
      del kwargs['use_params']
      self.model = MLPRegressor(**kwargs)
    self.scaler = StandardScaler()
    self.scaler_y = StandardScaler()

  def fit(self, X, y):
    X_scaled = self.scaler.fit_transform(X)
    y_train_2d = y.to_numpy().reshape(-1, 1)
    y_scaled = self.scaler_y.fit_transform(y_train_2d)
    self.model.fit(X_scaled, y_scaled.ravel())
    return self

  def predict(self, X):
    X_scaled = self.scaler.transform(X)
    return self.model.predict(X_scaled)

  def predict_proba(self, X):
    X_scaled = self.scaler.transform(X)
    return self.model.predict_proba(X_scaled)
