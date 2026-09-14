from pathlib import Path
import sys
# adding the project root to sys.path to ensure that src can be imported
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import typer
from loguru import logger
from pathlib import Path
from src.dataset_modules import LOADER_REGISTRY
import src.dataset_modules.semeval_loader

app = typer.Typer()

@app.command()

# function to rehydrate datasets (semeval)
def rehydrate(datasets: str = typer.Option(..., "--datasets", help="name of the dataset to rehydrate")):
    # check if the dataset is registered in the LOADER_REGISTRY
    if datasets not in LOADER_REGISTRY:
        logger.error(f"dataset '{datasets}' not registered")
        raise typer.Exit(code=1)
    # define the raw directory path for the dataset
    logger.info(f"starting rehydration for dataset: {datasets}")
    raw_dir = str(Path("data/01_raw") / datasets)
    # get the loader class from the registry and instantiate it
    loader_class = LOADER_REGISTRY[datasets]
    loader = loader_class(raw_dir=raw_dir)
    # call the acquire method to rehydrate the dataset
    loader.acquire()
    # check if the dataset is ready for processing
    if loader.check_readiness():
        logger.success(f"[{datasets}] rehydration completed successfully.")
    else:
        logger.error(f"[{datasets}] rehydration failed")

if __name__ == "__main__":
    app()