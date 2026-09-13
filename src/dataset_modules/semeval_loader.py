import os
import subprocess
import zipfile
from loguru import logger
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Any
import requests
from .base_loader import BaseDatasetLoader
from . import register_loader

@register_loader("semeval")
class SemevalLoader(BaseDatasetLoader):
    label_mapping = {
        "Yes": "conspiracy", 
        "No": "no_conspiracy",
        "Can't tell": "cant_tell"
    }
    def acquire(self) -> None:
            if self.check_readiness():
                logger.info("SemEval dataset already rehydrated. Skipping acquisition.")
                return
            raw_path = Path(self.raw_dir)
            os.makedirs(raw_path, exist_ok=True)
            # rehydration script
            script_path = raw_path / "rehydrate_data.py"
            # downloading the starter pack from GitHub and extracting it
            logger.info("downloading SemEval from GitHub")
            # url of the GitHub repository
            zip_url = "https://github.com/hide-ous/semeval26_task10_starter_pack/archive/refs/heads/main.zip"
            response = requests.get(zip_url)
            if response.status_code == 200:
                zip_path = raw_path / "github_repo.zip"
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                # extracting the zip file
                logger.info("extracting the GitHub zip file")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_path)
                # moving the contents of the extracted folder to the raw_dir
                extracted_folder = raw_path / "semeval26_task10_starter_pack-main"
                if extracted_folder.exists():
                    for item in extracted_folder.iterdir():
                        item.rename(raw_path / item.name)
                    extracted_folder.rmdir()
                # removing the zip file after extraction
                zip_path.unlink()
                logger.success("SemEval files downloaded and extracted successfully.")
            else:
                logger.error(f"failed to download SemEval from GitHub with status code: {response.status_code}")
                return
            # downloading the redacted and filtered test files from Zenodo
            zenodo_files = {
                "train_redacted.jsonl": "https://zenodo.org/records/15114172/files/train_redacted.jsonl?download=1",
                "dev_redacted.jsonl": "https://zenodo.org/records/15114172/files/dev_redacted.jsonl?download=1"
            }
            for filename, url in zenodo_files.items():
                filepath = raw_path / filename
                if not filepath.exists():
                    logger.info(f"downloading {filename} from Zenodo to {filepath}")
                    try:
                        res = requests.get(url)
                        if res.status_code == 200:
                            with open(filepath, 'wb') as f:
                                f.write(res.content)
                            logger.success(f"{filename} downloaded successfully from Zenodo")
                        else:
                            logger.warning(f"Zenodo responded with status code {res.status_code} for {filename}")
                    except Exception as e:
                        logger.warning(f"failed to contact Zenodo for {filename}: {e}")
            # checking if the required files are present in the raw_dir
            required_files = ["train_redacted.jsonl", "dev_redacted.jsonl"]
            missing_files = [f for f in required_files if not (raw_path / f).exists()]
            # if any required files are missing, log an error and provide instructions for manual download
            if missing_files:
                logger.error(
                    f"not possible to download all required files automatically.\n"
                    f"missing files: {missing_files}\n"
                    f"go to Zenodo: https://zenodo.org/doi/10.5281/zenodo.15114172\n"
                    f"download the missing files manually\n"
                    f"place them in the folder: {os.path.abspath(raw_path)} and rerun the script\n"
                )
                return
            # checking if the rehydration script exists and executing it
            if script_path.exists():
                logger.info("executing rehydration script")
                try:
                    subprocess.run(["python", "rehydrate_data.py"], cwd=self.raw_dir, check=True)
                    logger.success("rehydration of SemEval completed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"error occurred while executing the rehydration script: {e}")

    def check_readiness(self) -> bool:
        raw_path = Path(self.raw_dir)
        # checking for the existence of both redacted and rehydrated files for train and dev and the existence of the filtered test file
        train = (raw_path / "train_redacted.jsonl").exists() and (raw_path / "train_rehydrated.jsonl").exists()
        dev = (raw_path / "dev_redacted.jsonl").exists() and (raw_path / "dev_rehydrated.jsonl").exists()
        test = (raw_path / "test_filtered.jsonl").exists() 
        return train and dev and test

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        raw_path = Path(self.raw_dir)
        files_to_process = [
            ("train", "train"),
            ("dev", "val")
        ]
        # merging the redacted and rehydrated files for each of the train and dev splits
        for file_prefix, target_split in files_to_process:
            path_labels = raw_path / f"{file_prefix}_redacted.jsonl"
            path_texts = raw_path / f"{file_prefix}_rehydrated.jsonl"
            if not path_labels.exists() or not path_texts.exists():
                print(f"[SemEval] file {file_prefix} not found. Skipping...")
                continue
            # merging ids and annotated labels
            df_labels = pd.read_json(path_labels, lines=True)
            df_texts = pd.read_json(path_texts, lines=True)
            df_merged = pd.merge(df_labels, df_texts[['_id', 'text']], on="_id", how="inner")
            # iterating over the merged dataframe and yielding each record with harmonized fields
            for _, row in df_merged.iterrows():
                yield self._process_row(row, target_split)
        # processing the test split, which is already filtered and contains both _id and text
        test_path = raw_path / "test_filtered.jsonl"
        if test_path.exists():
            df_test = pd.read_json(test_path, lines=True)
            for _, row in df_test.iterrows():
                yield self._process_row(row, "test")
        else:
            print("[SemEval] file test_filtered.jsonl not found. Skipping test split...")

    # helper function to process each row of the dataframe and return a harmonized record
    def _process_row(self, row, target_split):
        example = row.to_dict()
        # collection of text markers
        markers_list = example.get('markers', [])
        themes = list(set(d.get('type') for d in markers_list if isinstance(d, dict) and 'type' in d))
        # collection of every marker and associated text span
        spans = []
        for m in markers_list:
            spans.append({
                'start_char': int(m.get('startIndex', 0)),
                'end_char': int(m.get('endIndex', 0)),
                'label': str(m.get('type', 'Unknown')),
                'span_text': str(m.get('text', ''))
            })
        # original label
        raw_label = example.get('conspiracy', "Can't tell")
        # mapping raw label to canonical label
        target_label = self.label_mapping.get(raw_label, "EXCLUDE")
        # final harmonized result
        return {
            'original_id': str(example.get('_id', '')),
            'text': example.get('text', ''),
            'canonical_label': target_label,
            'original_split': target_split,
            'metadata': {
                'themes': themes,
                'spans': spans, 
                'domain': 'Social Media',
                'source': 'SemEval'
            }
        }