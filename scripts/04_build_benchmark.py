import typer
import pandas as pd
from pathlib import Path
from loguru import logger
from datasets import Dataset
from src.harmonizer import HarmonizerValidator
from src.dataset_modules import LOADER_REGISTRY
import src.dataset_modules.ccn_loader
import src.dataset_modules.voat_loader
import src.dataset_modules.semeval_loader

app = typer.Typer()

def apply_splits(df_dataset: pd.DataFrame, dataset_name: str, save_dir: Path):
    # function to split the dataset into train, validation, and test sets (80/10/10) and save them as JSONL files.
    logger.info(f"[{dataset_name}] split creation 80/10/10...")
    hf_dataset = Dataset.from_pandas(df_dataset)
    train_test = hf_dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    # creating a dictionary to hold the splits and converting them to pandas DataFrames for saving
    splits = {
        "train": train_test['train'].to_pandas(),
        "val": val_test['train'].to_pandas(),
        "test": val_test['test'].to_pandas()
    }
    # creating the save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        file_path = save_dir / f"{dataset_name}_{split_name}.jsonl"
        split_df.to_json(file_path, orient="records", lines=True)

@app.command()
def build_benchmark():
    # function to build a unified benchmark dataset from all registered datasets
    # initialize the HarmonizerValidator with the canonical taxonomy
    validator = HarmonizerValidator("config/canonical_taxonomy.yaml")
    benchmark_dir = Path("data/03_benchmark/")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    logger.info("starting benchmark build process...")
    # iterating over all datasets registered in LOADER_REGISTRY to process and validate their raw data
    for dataset_name, loader_class in LOADER_REGISTRY.items():
        logger.info(f"processing {dataset_name}...")
        # defining the path to the raw data for the current dataset
        raw_dir = f"data/01_raw/{dataset_name}"
        loader = loader_class(raw_dir)
        if not loader.check_readiness():
            logger.warning(f"raw data not found for {dataset_name}. Make sure you have the files in {raw_dir}. Skipping...")
            continue
        # record validation, passing the records from the dataset to the validator to standardize and filter them
        valid_records = []
        # extracting data to validate it
        for record in loader.load_raw_records():
            validated = validator.process_and_validate(dataset_name, record)
            if validated:
                valid_records.append(validated)
        
        # saving valid records to a standardized JSONL file and then applying the train/val/test split
        if valid_records:
            logger.info(f"[{dataset_name}] {len(valid_records)} valid records found after harmonization.")
            df_valid = pd.DataFrame(valid_records)
            # saving the standardized dataset to a JSONL file in the data/02_standardized directory
            std_dir = Path("data/02_standardized")
            std_dir.mkdir(parents=True, exist_ok=True)
            df_valid.to_json(std_dir / f"{dataset_name}_standardized.jsonl", orient="records", lines=True)
            # applying the train/val/test split and saving the resulting files in the benchmark directory
            apply_splits(df_valid, dataset_name, benchmark_dir)
        else:
            logger.error(f"[{dataset_name}] no valid records found after harmonization. Check the raw data and the canonical taxonomy.")

    # after processing all datasets, we combine the individual train/val/test files into unified benchmark files for each split
    logger.info("combining all datasets into unified benchmark files...")
    for split in ["train", "val", "test"]:
        split_files = list(benchmark_dir.glob(f"*_{split}.jsonl"))
        if split_files:
            df_combined = pd.concat([pd.read_json(f, lines=True) for f in split_files], ignore_index=True)
            output_file = benchmark_dir / f"benchmark_{split}.jsonl"
            df_combined.to_json(output_file, orient="records", lines=True)
            for f in split_files:
                f.unlink()  
    logger.success(f"unified benchmark files generated in {benchmark_dir}")

if __name__ == "__main__":
    app()