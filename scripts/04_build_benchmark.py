import sys
from pathlib import Path
# adding the project root to sys.path to ensure that src can be imported
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import pandas as pd
from loguru import logger
import typer
from sklearn.model_selection import train_test_split
from enum import Enum
from src.dataset_modules import LOADER_REGISTRY
import src.dataset_modules.voat_loader
import src.dataset_modules.ccn_loader
import src.dataset_modules.semeval_loader
import src.dataset_modules.coco_loader
import src.dataset_modules.loco_loader
import src.dataset_modules.donald_loader
import src.dataset_modules.twitter_loader
from src.harmonizer import HarmonizerValidator

app = typer.Typer()

# different strategies for splitting the datasets into train/val/test
class SplitStrategy(str, Enum):
    preserve_original = "preserve_original_splits"
    leave_one_out = "leave_one_dataset_out"
    stratified = "stratified_re_split"

# split the dataset into train/val/test while preserving the original splits if they exist, otherwise fallback to 80/10/10 split
def apply_preserve_original_splits(df_valid: pd.DataFrame, dataset_name: str):
    # assigning all records to 'unassigned' if the loader did not provide an 'original_split' column
    if 'original_split' not in df_valid.columns:
        df_valid['original_split'] = 'unassigned'
    # splitting the dataset based on the 'original_split' column
    df_train_pre = df_valid[df_valid['original_split'] == 'train']
    df_val_pre = df_valid[df_valid['original_split'] == 'val']
    df_test_pre = df_valid[df_valid['original_split'] == 'test']
    # identifying records that do not have an original split assigned
    df_unassigned = df_valid[~df_valid['original_split'].isin(['train', 'val', 'test'])]
    # if there are unassigned records, we apply a fallback stratified 80/10/10 split
    if not df_unassigned.empty:
        logger.warning(f"[{dataset_name}] no original splits found, applying fallback stratified 80/10/10 split to {len(df_unassigned)} records...")
        # extracting the labels for stratification
        y_unassigned = df_unassigned['canonical_label']
        # first split, divide the unassigned records into 80% train and 20% temp (stratified based on y_unassigned)
        train_fall, temp_fall, _, y_temp = train_test_split(df_unassigned, y_unassigned, test_size=0.2, random_state=42, stratify=y_unassigned)
        # second split, divide the temp in half to get 10% val and 10% test (stratified based on y_temp)
        val_fall, test_fall = train_test_split(temp_fall, test_size=0.5, random_state=42, stratify=y_temp)
        # combining the pre-assigned splits with the fallback splits
        df_train = pd.concat([df_train_pre, train_fall]) if not df_train_pre.empty else train_fall
        df_val = pd.concat([df_val_pre, val_fall]) if not df_val_pre.empty else val_fall
        df_test = pd.concat([df_test_pre, test_fall]) if not df_test_pre.empty else test_fall
    else:
        df_train, df_val, df_test = df_train_pre, df_val_pre, df_test_pre
    # returning the final splits as lists of dictionaries
    return df_train.to_dict('records'), df_val.to_dict('records'), df_test.to_dict('records')

# splitting using a stratified re-split of the dataset into train/val/test, ignoring any original splits
def apply_stratified_re_split(df_valid: pd.DataFrame, dataset_name: str):
    logger.info(f"[{dataset_name}] applying 80/10/10 stratified re-split...")
    y = df_valid['canonical_label']
    # first split, divide the dataset into 80% train and 20% temp (stratified based on y)
    train_df, temp_df, _, y_temp = train_test_split(df_valid, y, test_size=0.2, random_state=42, stratify=y)
    # second split, divide the temp in half to get 10% val and 10% test (stratified based on y_temp)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=y_temp)
    # returning the final splits as lists of dictionaries
    return train_df.to_dict('records'), val_df.to_dict('records'), test_df.to_dict('records')

def apply_leave_one_out(df_valid: pd.DataFrame, dataset_name: str, target_dataset: str):
    # if the current dataset is the target dataset for leave-one-out, assign all records to test and leave train and val empty
    if dataset_name == target_dataset:
        logger.info(f"[{dataset_name}] leave-one-out strategy, all records assigned to test, train and val empty.")
        # returning empty train and val, and all records in test
        return [], [], df_valid.to_dict('records')
    # if the current dataset is not the target dataset, assign all records to train and val, and leave test empty
    else:
        logger.info(f"[{dataset_name}] other dataset in leave-one-out strategy, all records assigned to train and val, test empty.")
        y = df_valid['canonical_label']
        train_df, val_df = train_test_split(df_valid, test_size=0.1, random_state=42, stratify=y)
        # returning the final splits as lists of dictionaries, with test empty
        return train_df.to_dict('records'), val_df.to_dict('records'), []

# main function to build the benchmark dataset
@app.command()
def build_benchmark(split_strategy: SplitStrategy = typer.Option(SplitStrategy.preserve_original, "--split-strategy",  help="specifies the strategy for splitting datasets into train/val/test."),
    target_dataset: str = typer.Option(None, "--target-dataset", help="specifies the target dataset for leave-one-out strategy. Required if using leave_one_out strategy.")):
    logger.info(f"starting benchmark build with strategy: {split_strategy.value}")
    # validating that target_dataset is specified if using leave_one_out strategy
    if split_strategy == SplitStrategy.leave_one_out and not target_dataset:
        logger.error("target_dataset must be specified when using leave_one_out strategy.")
        raise typer.Exit(1)
    # initializing the validator and creating necessary directories
    validator = HarmonizerValidator("config/canonical_taxonomy.yaml")
    benchmark_dir = Path("data/03_benchmark")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    std_dir = Path("data/02_standardized")
    std_dir.mkdir(parents=True, exist_ok=True)
    # initializing lists to hold the final train, val, and test records
    train = []
    val = []
    test = []
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
        else:
            logger.error(f"[{dataset_name}] no valid records found after harmonization. Check the raw data and the canonical taxonomy.")

        # applying the specified split strategy to the validated records and extending the final train, val, and test lists accordingly
        # preserving the original splits if they exist, otherwise applying a fallback split
        if split_strategy == SplitStrategy.preserve_original:
            train_recs, val_recs, test_recs = apply_preserve_original_splits(df_valid, dataset_name)
            train.extend(train_recs)
            val.extend(val_recs)
            test.extend(test_recs)
        # applying leave-one-out strategy, where the target dataset is used as the test set and all other datasets are used for training and validation
        elif split_strategy == SplitStrategy.leave_one_out:
            train_recs, val_recs, test_recs = apply_leave_one_out(df_valid, dataset_name, target_dataset)
            train.extend(train_recs)
            val.extend(val_recs)
            test.extend(test_recs)
        # applying a stratified re-split of the dataset into train/val/test, ignoring any original splits
        elif split_strategy == SplitStrategy.stratified:
            train_recs, val_recs, test_recs = apply_stratified_re_split(df_valid, dataset_name)
            train.extend(train_recs)
            val.extend(val_recs)
            test.extend(test_recs)

    # saving the final combined benchmark datasets to JSONL files in the data/03_benchmark directory
    logger.info("saving benchmark datasets to JSONL files...")
    pd.DataFrame(train).to_json(benchmark_dir / "benchmark_train.jsonl", orient="records", lines=True)
    pd.DataFrame(val).to_json(benchmark_dir / "benchmark_val.jsonl", orient="records", lines=True)
    pd.DataFrame(test).to_json(benchmark_dir / "benchmark_test.jsonl", orient="records", lines=True)
    logger.success(f"benchmark datasets saved to {benchmark_dir}")

if __name__ == "__main__":
    app()