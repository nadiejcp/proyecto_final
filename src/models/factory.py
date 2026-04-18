from .baseline import BaselineModel
from .random_forest import RandomForestModel
from .mlp import MLPModel
from .xgb_boost import XGBoostModel
from .gradient_boosting import GradientBoostingModel
from .decision_tree import DecisionTreeModel
from .sgd import SGDRegressorModel

class ModelFactory:
  def __init__(self, config: dict):
    self.config = config

  def list_models(self):
    """Returns the list of model names defined in the configuration."""
    return list(self.config.keys())

  def create(self, name: str):
    """Creates and returns a model instance based on the configuration name."""
    if name not in self.config:
      raise ValueError(f"Model configuration '{name}' not found in config.")

    model_conf = self.config[name].copy()
    model_type = model_conf.pop("type")

    if model_type == "linear_regression":
      return BaselineModel(**model_conf)
    elif model_type == "random_forest":
      return RandomForestModel(**model_conf)
    elif model_type == "mlp":
      return MLPModel(hidden_layers=(100, 100, 100), **model_conf)
    elif model_type == "xgboost":
      return XGBoostModel(**model_conf)
    elif model_type == "gradient_boosting":
      return GradientBoostingModel(**model_conf)
    elif model_type == "decision_tree":
      return DecisionTreeModel(**model_conf)
    elif model_type == "sgd_regressor":
      return SGDRegressorModel(**model_conf)
    else:
      raise ValueError(f"Unknown model type: {model_type}")
