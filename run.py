# run.py

from pipeline_controller import PipelineController
from config.load_config import load_config
from utils.seed import set_global_seed

def main():
  config = load_config("config/config.yaml")
  set_global_seed(config["seed"])
  pipeline = PipelineController(config)
  pipeline.run()

if __name__ == "__main__":
  main()
