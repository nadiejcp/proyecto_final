"""
GridSearchTuner
==============
Uses ModelFactory to instantiate a model and then runs sklearn's GridSearchCV
to find the best hyperparameters for that model.

Usage (from project root):
  from src.models.factory import ModelFactory
  from models.grid_search import GridSearchTuner

  factory = ModelFactory(config["models"])
  tuner   = GridSearchTuner(factory, config)
  result  = tuner.tune("random_forest", X_train, y_train)
  print(result)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

from utils.functions import load_dataset, save_model

PARAM_GRIDS: dict[str, dict] = {
  "linear_regression": {
    "fit_intercept": [True, False],
  },
  "random_forest": {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 1.0],
  },
  "xgboost": {
    "n_estimators": [300, 500, 1000],
    "max_depth": [6, 7],
    "learning_rate": [0.01, 0.03, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "reg_alpha": [0, 0.5, 1.0],
    "gamma": [0, 0.03, 0.05, 0.07, 0.1],
    "reg_lambda": [1.0, 5.0, 10.0],
    'objective':['reg:squarederror'],
  },
  "decision_tree": {
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
    "splitter": ["best", "random"],
  },
  "sgd_regressor": {
    "loss": ["squared_error", "huber", "epsilon_insensitive"],
    "penalty": ["l2", "l1", "elasticnet"],
    "alpha": [0.0001, 0.001, 0.01],
    "max_iter": [1000, 2000],
    "learning_rate": ["invscaling", "optimal", "constant", "adaptive"],
    "eta0": [0.01, 0.1],
  },
  "gradient_boosting": {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
  },
}

rmse_scorer = make_scorer(
  lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
  greater_is_better=False,
)

class GridSearchTuner:
  """
  Runs GridSearchCV for a named model (from the config) and returns
  the best parameters and best cross-validated RMSE.

  Parameters
  ----------
  model_factory : ModelFactory
      An already-instantiated ModelFactory.
  config : dict
      The full project config (used to look up model type).
  cv : int
      Number of cross-validation folds (default 5).
  n_jobs : int
      Parallel jobs for GridSearchCV (default -1 = all CPUs).
  verbose : int
      Verbosity level for GridSearchCV.
  """

  def __init__(self, model_factory, config: dict, cv: int = 5,
               n_jobs: int = -1, verbose: int = 2):
    self.model_factory = model_factory
    self.config = config
    self.cv = cv
    self.n_jobs = n_jobs
    self.verbose = verbose

  def tune(self, model_name: str) -> dict:
    """
    Perform GridSearchCV for *model_name* and return a result dict.

    Parameters
    ----------
    model_name : str
        Key that exists in config["models"], e.g. ``"random_forest"``.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.

    Returns
    -------
    dict with keys:
        - ``model_name``    : the name that was tuned
        - ``best_params``   : dict with the winning hyperparameters
        - ``best_rmse``     : best (positive) RMSE from CV
        - ``cv_results``    : full cv_results_ from GridSearchCV
        - ``best_estimator``: fitted sklearn estimator with best params
    """
    X_train, y_train = load_dataset(self.config["data"]["interim_data_path"] + "/train_set.csv", self.config["features"]["target"])
    model_type = self._get_model_type(model_name)
    estimator  = self._get_raw_estimator(model_name)
    param_grid = self._get_param_grid(model_type)

    print(f"\n{'='*60}")
    print(f"  Grid Search  →  {model_name}  ({model_type})")
    print(f"  Param grid size: {self._grid_size(param_grid)} combinations")
    print(f"{'='*60}\n")

    grid_cv = GridSearchCV(
      estimator=estimator,
      param_grid=param_grid,
      scoring=rmse_scorer,
      cv=self.cv,
      n_jobs=self.n_jobs,
      verbose=self.verbose,
      refit=True,
      return_train_score=False,
    )
    grid_cv.fit(X_train, y_train)

    best_rmse = -grid_cv.best_score_   # flip sign back to positive RMSE

    print(f"\n✓  Best RMSE  : {best_rmse:,.4f}")
    print(f"✓  Best Params: {grid_cv.best_params_}\n")

    save_model(grid_cv.best_estimator_, self.config['best_model']['save_path'])

    return {
        "model_name":     model_name,
        "best_params":    grid_cv.best_params_,
        "best_rmse":      best_rmse,
        "cv_results":     grid_cv.cv_results_,
        "best_estimator": grid_cv.best_estimator_,
    }

  def tune_all(self, X_train, y_train) -> dict:
    """
    Run ``tune()`` for every model listed in config["models"].

    Returns
    -------
    dict  {model_name: tune_result_dict}
    """
    results = {}
    for model_name in self.model_factory.list_models():
      try:
        results[model_name] = self.tune(model_name, X_train, y_train)
      except Exception as exc:
        print(f"[WARN] Could not tune '{model_name}': {exc}")
    return results

  def best_overall(self, results: dict) -> dict:
    """
    Given the dict returned by ``tune_all``, return the entry with the
    lowest RMSE.
    """
    return min(results.values(), key=lambda r: r["best_rmse"])

  def _get_model_type(self, model_name: str) -> str:
    """Reads the ``type`` field from config["models"][model_name]."""
    models_cfg = self.config.get("models", {})
    if model_name not in models_cfg:
      raise ValueError(
        f"Model '{model_name}' not found in config['models']. "
        f"Available: {list(models_cfg.keys())}"
      )
    return models_cfg[model_name]["type"]

  def _get_raw_estimator(self, model_name: str):
    """
    Uses the factory to build the wrapper, then extracts the underlying
    sklearn estimator so GridSearchCV can work with it directly.
    """
    wrapper = self.model_factory.create(model_name)
    if not hasattr(wrapper, "model"):
      raise AttributeError(
        f"Model wrapper for '{model_name}' has no `.model` attribute. "
        "Make sure all model classes store the sklearn estimator as "
        "`self.model`."
      )
    return type(wrapper.model)()

  def _get_param_grid(self, model_type: str) -> dict:
    """Returns the param grid for the given model type."""
    if model_type not in PARAM_GRIDS:
      raise ValueError(
        f"No param grid defined for model type '{model_type}'. "
        f"Available types: {list(PARAM_GRIDS.keys())}"
      )
    return PARAM_GRIDS[model_type]

  @staticmethod
  def _grid_size(param_grid: dict) -> int:
    """Counts the total number of hyperparameter combinations."""
    total = 1
    for values in param_grid.values():
      total *= len(values)
    return total
