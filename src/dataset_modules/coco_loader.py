import os
from pathlib import Path
from typing import Generator, Dict, Any
from loguru import logger
from .base_loader import BaseDatasetLoader
from . import register_loader

@register_loader("coco")
class CocoLoader(BaseDatasetLoader):
    label_mapping = {
        "CONSPIRACY": "conspiracy",
        "UNRELATED": "no_conspiracy",
        "RELATED": "no_conspiracy" 
    }

    COCO_CATEGORIES = [
        "Suppressed cures", "Mind Control", "Antivax", "Fake virus",
        "Intentional Pandemic", "Harmful Radiation_Influence",
        "Population reduction_control", "New World Order",
        "Esoteric Misinformation", "Satanism", "Other Conspiracy",
        "Other non-Conspiracy Misinformation"
    ]

    def acquire(self) -> None:
        if self.check_readiness():
            logger.info("COCO dataset already present, skipping acquisition.")
            return
        # create the raw directory if it doesn't exist
        raw_path = Path(self.raw_dir)
        os.makedirs(raw_path, exist_ok=True)
        logger.error(
            f"COCO is a licensed dataset (EU Digital Services Act, Art. 40)\n"
            f"all the instructions are available at: https://huggingface.co/datasets/Jlangguth/COCO\n"
            f"after approval, place the received files in: {os.path.abspath(raw_path)}\n"
            f"then rerun the script\n"
        )

    def check_readiness(self) -> bool:
        return (Path(self.raw_dir) / "COCO_export").exists()

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        file_path = Path(self.raw_dir) / "COCO_export"
        # reading the COCO dataset
        with open(file_path, "r", encoding="utf-8") as f:
            # iterating over each line in the file
            for line in f:
                if len(line.strip()) <= 10:
                    continue
                # splitting the line into parts to extract tweet ID, scores, and text
                parts = line.split()
                tweet_id = parts[0]
                scores = parts[1:13]
                themes = []
                for i in range(12):
                    # if the category is classified as a conspiracy or related i save it as the theme of the text
                    if i < len(scores) and scores[i] in ['3', '2']:
                        # formatting the theme by replacing spaces and slashes with underscores to ensure consistency in naming
                        formatted_theme = self.COCO_CATEGORIES[i].replace(" ", "_").replace("/", "_")
                        themes.append(formatted_theme)
                # extracting the remaining words in the line as the text content of the tweet
                remaining_words = parts[13:]
                raw_label = "EXCLUDE"
                text_start_idx = 0
                # searching for global evaluation and start of the text
                for i, word in enumerate(remaining_words):
                    if word in ['CONSPIRACY', 'UNRELATED', 'RELATED']:
                        raw_label = word
                         # start of the text
                        text_start_idx = i + 1
                        break
                # reconstructing the text 
                clean_text = " ".join(remaining_words[text_start_idx:])
                # mapping the raw label to a canonical label using the predefined label mapping
                target_label = self.label_mapping.get(raw_label, "EXCLUDE")
                 # final harmonized result
                yield {
                    'original_id': str(tweet_id),
                    'text': clean_text,
                    'canonical_label': target_label,
                    'original_split': 'unassigned',
                    'metadata': {
                        'themes': themes,
                        'spans': [],
                        'domain': 'Social Media',
                        'source': 'COCO'
                    }
                }