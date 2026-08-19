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
        return (raw_path / "train_redacted.jsonl").exists() and (raw_path / "train_rehydrated.jsonl").exists()

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        path_labels = Path(self.raw_dir) / "train_redacted.jsonl"
        path_texts = Path(self.raw_dir) / "train_rehydrated.jsonl"
        # merging ids and annotated labels
        df_labels = pd.read_json(path_labels, lines=True)
        df_texts = pd.read_json(path_texts, lines=True)
        df_merged = pd.merge(df_labels, df_texts[['_id', 'text']], on="_id", how="inner")
        for _, row in df_merged.iterrows():
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
            yield {
                'original_id': str(example.get('_id', '')),
                'text': example.get('text', ''),
                'canonical_label': target_label,
                'original_split': 'unassigned',
                'metadata': {
                    'themes': themes,
                    'spans': spans, 
                    'domain': 'Social Media',
                    'source': 'SemEval'
                }
            }