import json
from pathlib import Path
from typing import Generator, Dict, Any
from .base_loader import BaseDatasetLoader
from . import register_loader

@register_loader("loco")
class LocoLoader(BaseDatasetLoader):
    label_mapping = {
        "conspiracy": "conspiracy",
        "mainstream": "no_conspiracy",
    }

    def acquire(self) -> None:
        pass

    def check_readiness(self) -> bool:
        p = Path(self.raw_dir)
        return (p / "data/LOCO.json").exists() and (p / "data/topic_description.json").exists()

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        p = Path(self.raw_dir)
        loco_topic_map = {}
        # mapping topic IDs to their corresponding first 4 words from the topic description file
        with open(p / "data/topic_description.json", 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
            for item in topics_data:
                topic_id = item.get("Topic_ID")
                top_words = item.get("Top_15_words", "")
                if topic_id and top_words:
                    # splitting the string by commas, removing the spaces, taking only the first 4 words and joining them with underscores.
                    first_4_words = [word.strip() for word in top_words.split(",")[:4]]
                    # creating a mapping from topic_id to the first 4 words joined by underscores to show that it is a single topic
                    loco_topic_map[topic_id] = "_".join(first_4_words)

        # reading the LOCO dataset and yielding each record with harmonized fields
        with open(p / "data/LOCO.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for example in data:
                themes = []
                # the three LDA topic keys in the dataset
                # collectingfor each key the associated topic id and searching for the corrisponding textual themes in the dictionary
                for key in ['topic_k100', 'topic_k200', 'topic_k300']:
                    raw_topic_id = example.get(key)
                    if raw_topic_id and raw_topic_id in loco_topic_map:
                        themes.append(loco_topic_map[raw_topic_id])
                # original label
                raw_label = example.get('subcorpus', "EXCLUDE")
                # mapping raw label to canonical label
                target_label = self.label_mapping.get(raw_label, "EXCLUDE")
                # final harmonized result
                yield {
                    'original_id': str(example.get('doc_id', '')),
                    'text': example.get('txt', ''),
                    'canonical_label': target_label,
                    'original_split': 'unassigned',
                    'metadata': {
                        'themes': themes,
                        'spans': [],
                        'domain': 'News',
                        'source': 'LOCO'
                    }
                }