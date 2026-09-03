import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Any
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
        pass

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