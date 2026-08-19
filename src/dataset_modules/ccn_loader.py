import json
from pathlib import Path
from typing import Generator, Dict, Any
from .base_loader import BaseDatasetLoader
from . import register_loader

@register_loader("ccn")
class CCNLoader(BaseDatasetLoader):
    #label mapping
    label_mapping = {
        "CONSPIRACY": "conspiracy", 
        "CRITICAL": "no_conspiracy",
    }
    
    def acquire(self) -> None:
        pass

    def check_readiness(self) -> bool:
        return (Path(self.raw_dir) / "dataset_en_gold.json").exists()

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        file_path = Path(self.raw_dir) / "dataset_en_gold.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for example in data:
            spans = []
            themes = set()
        # collection of themes and markers in the annotation section
            for a in example.get('annotations', []):
                span_label = str(a.get('category', 'Unknown'))
                themes.add(span_label)
                spans.append({
                    'start_char': int(a.get('start_char', 0)),
                    'end_char': int(a.get('end_char', 0)),
                    'label': span_label,
                    'span_text': str(a.get('span_text', ''))
                })
            # original label
            raw_label = example.get('category', "Can't tell")
            # mapping raw label to canonical label
            target_label = self.label_mapping.get(raw_label, "EXCLUDE")
            # final harmonized result
            yield {
                'original_id': str(example.get('id', '')),
                'text': example.get('text', ''),
                'canonical_label': target_label,
                'original_split': 'unassigned', 
                'metadata': {
                    'themes': list(themes),
                    'spans': spans,
                    'domain': 'Social Media',
                    'source': 'XAI-DisInfodemics'
                }
            }