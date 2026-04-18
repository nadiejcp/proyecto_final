# run.py

from pipeline_controller import PipelineController
from config.load_config import load_config
from utils.seed import set_global_seed

import uvicorn

def main():
  config = load_config("config/config.yaml")
  set_global_seed(config["seed"])
  print(' Seleccione una accion: ')
  print(' 1. Entrenar al modelo (recomendable si es la primera vez en ejecutar)')
  print(' 2. Despegar servidor')
  action = input('Ingrese una opcion: ')
  if '1' in action:
    pipeline = PipelineController(config)
    pipeline.run()
  elif '2' in action:
    port = 8000
    print(f"Starting server on port {port}")
    uvicorn.run(
      app='src.api.main:app',
      host='127.0.0.1',
      port=port,
      reload=True,
      reload_excludes=["venv"],
    )

if __name__ == "__main__":
  main()
