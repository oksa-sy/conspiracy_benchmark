import json
import os
from pathlib import Path
from typing import Generator, Dict, Any

from loguru import logger
import requests
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
        # method to acquire the dataset, in this case it will download the dataset from a given URL and save it to the raw_dir
        url = "https://raw.githubusercontent.com/dkorenci/oppositional-discourse/main/dataset/dataset_en_gold.json"
        os.makedirs(self.raw_dir, exist_ok=True)
        # extract the original filename from the URL
        filename = url.split('/')[-1]
        # define the filepath where the file will be saved
        filepath = os.path.join(self.raw_dir, filename)
        # check if the file already exists, if not download it
        if not os.path.exists(filepath):
            logger.info(f"downloading {filename} from {url} to {filepath}")
            response = requests.get(url)
            # check if the request was successful
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.success(f"completed downloading {filepath}")
            else:
                logger.error(f"error {response.status_code} occurred while downloading from the repository.")
        else:
            logger.info(f"{filename} already exists in {self.raw_dir}, skipping download")

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