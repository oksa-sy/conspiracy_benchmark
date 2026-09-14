import os
from loguru import logger
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Any
import requests
from .base_loader import BaseDatasetLoader
from . import register_loader

@register_loader("voat")

class VoatLoader(BaseDatasetLoader):
    label_mapping = {
        1: "conspiracy", 
        0: "no_conspiracy",
    }

    def acquire(self) -> None:
        # method to acquire the dataset, in this case it will download the dataset from a given URL and save it to the raw_dir
        url = "https://raw.githubusercontent.com/nika-akin/LLMs-and-conspiracy-beliefs/main/data/voat_annotation.csv"
        data_dir = os.path.join(self.raw_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        # extract the original filename from the URL
        filename = url.split('/')[-1]
        # define the filepath where the file will be saved
        filepath = os.path.join(data_dir, filename)
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
                logger.error(f"error {response.status_code} occurred while downloading from the repository")
        else:
            logger.info(f"{filename} already exists in {self.raw_dir}, skipping download")

    def check_readiness(self) -> bool:
        return (Path(self.raw_dir) / "data/voat_annotation.csv").exists()

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        file_path = Path(self.raw_dir) / "data/voat_annotation.csv"
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            example = row.to_dict()
            column_themes = ['Actor', 'Action', 'Threat', 'Pattern', 'Secrecy']
            active_themes = []
            spans = []
            text = str(example.get('body', ""))
            for theme in column_themes:
                    # if the theme is true I add it to the list of themes and collect the associated text span
                    if example.get(theme) is True:
                        active_themes.append(theme)
                        span_col_name = f"{theme}_span"
                        span_text = str(example.get(span_col_name, "")).strip()
                        # if there is a span text, i search for its position in the text
                        if span_text and span_text != "NA":
                            start_char = text.find(span_text)
                            if start_char != -1:
                                spans.append({
                                    'start_char': start_char,
                                    'end_char': start_char + len(span_text),
                                    'label': theme,
                                    'span_text': span_text
                                })
            # original id
            doc_id = example.get('comment_id') if pd.notna(example.get('comment_id')) else example.get('submission_id')
            # original label
            raw_label = example.get('overall_ct', "Can't tell")
            # mapping raw label to canonical label
            target_label = self.label_mapping.get(raw_label, "EXCLUDE")
            # final harmonized result
            yield {
                'original_id': str(doc_id),
                'text': text,
                'canonical_label': target_label,
                'original_split': 'unassigned',
                'metadata': {
                    'themes': active_themes,
                    'spans': spans,
                    'domain': 'Social Media',
                    'source': 'Voat'
                }
            }