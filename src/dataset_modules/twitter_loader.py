import os
from loguru import logger
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Any
from .base_loader import BaseDatasetLoader
from . import register_loader

@register_loader("twitter_conspiracy")
class TwitterConspiracyLoader(BaseDatasetLoader):
    # label mapping
    label_mapping = {
        1: "conspiracy", 
        0: "no_conspiracy"
    }
    # mapping of CMD theme columns to readable names
    cmd_themes = {
        'cmd.agenda.centroid.1': 'Agenda',
        'cmd.threat.centroid.1': 'Threat',
        'cmd.coalition.centroid.1': 'Coalition',
        'cmd.propaganda.centroid.1': 'Propaganda',
        'cmd.great_awakening.centroid.1': 'Great Awakening'
    }

    def acquire(self) -> None:
        if self.check_readiness():
            logger.info("Twitter Conspiracy dataset already present, skipping acquisition")
            return
        # create the raw directory if it doesn't exist
        raw_path = Path(self.raw_dir)
        os.makedirs(raw_path, exist_ok=True)
        logger.error(
            f"a formal request is required to obtain the Twitter Conspiracy dataset\n"
            f"contact the authors of the paper (e.g., Veronika Batzdorfer: Veronika.Batzdorfer@gesis.org)\n"
            f"place the received files in the folder: {os.path.abspath(raw_path)}\n"
            f"then rerun the script\n"
        )

    def check_readiness(self) -> bool:
        # check if all required CSV files exist in the expected directories
        p = Path(self.raw_dir)
        texts_path = p / "timelines" / "filteredconspiracnobots.csv"
        control_path = p / "timelines" / "filteredcontrol.csv"
        labels_path = p / "cmd" / "target_ct_binary_twt_cmd.csv"
        return texts_path.exists() and control_path.exists() and labels_path.exists()

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        p = Path(self.raw_dir)
        texts_path = p / "timelines" / "filteredconspiracnobots.csv"
        control_path = p / "timelines" / "filteredcontrol.csv"
        labels_path = p / "cmd" / "target_ct_binary_twt_cmd.csv"
        # load the CSV files and separate positive (conspiracy) and negative (control) samples
        df_texts = pd.read_csv(texts_path, dtype=str)
        df_labels = pd.read_csv(labels_path, dtype=str)
        df_merged = pd.merge(df_texts, df_labels, left_on="status_id", right_on="docs", how="inner")
        # identify rows that have at least one CMD theme marked as '1' (conspiracy)
        theme_cols = list(self.cmd_themes.keys())
        is_conspiracy = df_merged[theme_cols].isin(['1', 1]).any(axis=1)
        # split/load positive samples and load negative samples from the control file
        df_pos = df_merged[is_conspiracy]
        df_neg = pd.read_csv(control_path, dtype=str)
        # limit the number of rows to a maximum of 10,000 for both positive and negative samples
        MAX_ROWS = 10000
        if len(df_pos) > MAX_ROWS:
            df_pos = df_pos.sample(n=MAX_ROWS, random_state=42)
        if len(df_neg) > MAX_ROWS:
            df_neg = df_neg.sample(n=MAX_ROWS, random_state=42)
        # combine the positive and negative samples and shuffle the resulting DataFrame
        df_final = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)
        # iterate over the final DataFrame and yield each record in the expected format
        for _, row in df_final.iterrows():
            active_themes = []
            text = str(row.get('text', ""))
            # for each CMD theme column, check if the value is '1' (indicating the presence of that theme) and collect the corresponding theme names and spans
            for col, theme_name in self.cmd_themes.items():
                if str(row.get(col, "0")).strip() == "1":
                    active_themes.append(theme_name)
            # determine the raw label based on the presence of active themes (1 for conspiracy, 0 for no_conspiracy)
            raw_label = 1 if len(active_themes) > 0 else 0
            target_label = self.label_mapping.get(raw_label, "EXCLUDE")
            # yield the final harmonized result
            yield {
                'original_id': str(row.get('status_id', '')),
                'text': text,
                'canonical_label': target_label,
                'original_split': 'unassigned',
                'metadata': {
                    'themes': active_themes,
                    'spans': [],
                    'domain': 'Social Media',
                    'source': 'Twitter_Conspiracy'
                }
            }