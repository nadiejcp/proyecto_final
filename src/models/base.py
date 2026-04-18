from abc import ABC, abstractmethod

class BaseModel(ABC):
  """
  Abstract base class for all models in the pipeline.
  """
  @abstractmethod
  def fit(self, X, y):
    """
    Train the model.
    """
    pass

  @abstractmethod
  def predict(self, X):
    """
    Make predictions.
    """
    pass

  @abstractmethod
  def predict_proba(self, X):
    """
    Predict probabilities.
    """
    pass
