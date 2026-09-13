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
import src.dataset_modules.voat_loader
import src.dataset_modules.ccn_loader
import src.dataset_modules.loco_loader
app = typer.Typer()

# list of public datasets to be downloaded
PUBLIC_DATASETS = ["voat", "ccn", "loco"]

@app.command()
# function to download public datasets
def download_public():
    for dataset_name in PUBLIC_DATASETS:
        if dataset_name in LOADER_REGISTRY:
            logger.info(f"public dataset: {dataset_name}")
            # define the raw directory path for the dataset
            raw_dir = str(Path("data/01_raw") / dataset_name)
            # get the loader class from the registry and instantiate it
            loader_class = LOADER_REGISTRY[dataset_name]
            loader = loader_class(raw_dir=raw_dir)
            # call the acquire method to download the dataset
            loader.acquire()
            # check if the dataset is ready for processing
            if loader.check_readiness():
                logger.success(f"[{dataset_name}] ready for processing")
            else:
                logger.error(f"[{dataset_name}] error during acquisition")
        # if the dataset is not found in the registry, print a warning message
        else:
            logger.warning(f"loader for {dataset_name} not found in the registry")

if __name__ == "__main__":
    app()