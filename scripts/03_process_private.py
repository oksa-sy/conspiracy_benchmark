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
import src.dataset_modules.coco_loader
import src.dataset_modules.donald_loader
import src.dataset_modules.twitter_loader

app = typer.Typer()

# list of private datasets to be processed
PRIVATE_DATASETS = ["coco", "twitter_conspiracy", "donald"]

@app.command()

# function to process private datasets
def process_private():
    logger.info("starting validation of private datasets")
    # check if each private dataset is registered in the LOADER_REGISTRY
    for dataset_name in PRIVATE_DATASETS:
        if dataset_name in LOADER_REGISTRY:
            # define the raw directory path for the dataset
            raw_dir = str(Path("data/01_raw") / dataset_name)
            # get the loader class from the registry and instantiate it
            loader_class = LOADER_REGISTRY[dataset_name]
            loader = loader_class(raw_dir=raw_dir)
            # this method will not download the dataset, but will check if the necessary files are present
            # and if not it will log istructions for the user to download them manually
            loader.acquire()
            # check if the dataset is ready for processing
            if loader.check_readiness():
                logger.info(f"[{dataset_name}] ready for processing")
            else:
                logger.warning(f"[{dataset_name}] skipping processing due to missing files. Please check the logs for instructions on how to download them")
        # if the dataset is not found in the registry, print a warning message
        else:
            logger.warning(f"loader for {dataset_name} not found in the registry")

if __name__ == "__main__":
    app()