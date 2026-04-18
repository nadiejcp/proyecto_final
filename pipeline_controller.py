from src.data.make_dataset import DataLoader
from src.data.split_data import split_and_save_data
from utils.functions import load_dataset
from src.models.factory import ModelFactory
from src.models.train_model import Trainer
from src.models.grid_search import GridSearchTuner
import pandas as pd

class PipelineController:
  def __init__(self, config: dict):
    self.config = config
    self.raw_data = None

  def run(self):
    data_loader = DataLoader(self.config)
    data_loader.fetch_data()
    data_loader.transform()
    split_and_save_data(self.config)
    model_factory = ModelFactory(self.config["models"])
    trainer = Trainer(self.config, model_factory)
    results = {}
    for model_name in model_factory.list_models():
      print(f'Training {model_name}...')
      trainer.train_best_model(model_name)
      print(f'Evaluating {model_name}...')
      metrics = trainer.evaluate_model(model_name, "/test_set.csv")
      results[model_name] = metrics
      print(f'Finished {model_name}')

    best_model = min(results, key=lambda x: results[x]["RMSE"])
    print(f'Best model: {best_model}')
    print(f'Best model metrics: {results[best_model]}')

    results_df = pd.DataFrame(results).T
    results_df.to_csv(self.config["data"]["path"] + "/results.csv")
    tuner = GridSearchTuner(model_factory, self.config, cv=5, n_jobs=-1)
    result = tuner.tune(best_model)
