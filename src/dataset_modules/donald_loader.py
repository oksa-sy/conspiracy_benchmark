import pandas as pd
import pyreadr
from pathlib import Path
from typing import Generator, Dict, Any
from .base_loader import BaseDatasetLoader
from . import register_loader

@register_loader("donald")
class DonaldLoader(BaseDatasetLoader):
    label_mapping = {
        True: "conspiracy",
        False: "no_conspiracy"
    }

    def acquire(self) -> None:
        pass

    def check_readiness(self) -> bool:
        p = Path(self.raw_dir)
        meta = p / "osfstorage-archive/DONALD/DONALD.rdata"
        files = [
            p / "dataverse_files/DONALD.txt.CT.rdata",
            p / "dataverse_files/DONALD.txt.LB.rdata",
            p / "dataverse_files/DONALD.txt.LL.rdata",
            p / "dataverse_files/DONALD.txt.RR.rdata"
        ]
        return meta.exists() and all(f.exists() for f in files)

    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        p = Path(self.raw_dir)
        meta_path = p / "osfstorage-archive/DONALD/DONALD.rdata"
        # uploading file with metadata
        meta_result = pyreadr.read_r(str(meta_path))
         # pyreadr returns a dict, dynamically grab the first loaded dataset regardless of its R object name
        df_meta = meta_result[list(meta_result.keys())[0]]
        # only keeping essential values
        useful_col = ['doc_id', 'domain_group', 'k100', 'k200', 'k300']
        present_col = [c for c in useful_col if c in df_meta.columns]
        df_meta = df_meta[present_col]
        files = [
            p / "dataverse_files/DONALD.txt.CT.rdata",
            p / "dataverse_files/DONALD.txt.LB.rdata",
            p / "dataverse_files/DONALD.txt.LL.rdata",
            p / "dataverse_files/DONALD.txt.RR.rdata"
        ]
         # setting a maximum of 5000 rows for each file to have an equal split and not overload memory
        MAX_ROWS = 5000
         # iterating over the 4 files in the dataset
        for file in files:
            try:
                # collecting texts and ids
                result = pyreadr.read_r(file)
                # pyreadr returns a dictionary, grabbing the first key dynamically because the R object name changes per file
                key = list(result.keys())[0]
                df_texts = result[key]
                # randomly collecting 5000 rows
                if len(df_texts) > MAX_ROWS:
                    # undersample the dataset, random_state=42 ensures the exact same sample every time
                    df_texts = df_texts.sample(n=MAX_ROWS, random_state=42)
                # linking text and IDs to metadata
                df_merged = pd.merge(df_texts, df_meta, on='doc_id', how='inner')
                # iterating over each row in the merged dataframe to yield harmonized records
                for _, row in df_merged.iterrows():
                    example = row.to_dict()
                    # label (conspiracy or not)
                    domain_group = str(example.get('domain_group', '')).strip().upper()
                    doc_id = str(example.get('doc_id', '')).strip().upper()
                    # if the domain group or id starts with 'CT' it is a conspiracy
                    is_conspiracy = domain_group.startswith('CT') or doc_id.startswith('CT')
                    target_label = self.label_mapping[is_conspiracy]
                    themes = []
                    # LDA topics set as themes
                    for k_col in ['k100', 'k200', 'k300']:
                        val = example.get(k_col)
                        if pd.notna(val) and str(val).lower() not in ["none", "nan"]:
                            themes.append(str(val))
                    # final harmonized result   
                    yield {
                        'original_id': str(example.get('doc_id', '')),
                        'text': str(example.get('txt', '')),
                        'canonical_label': target_label,
                        'original_split': 'unassigned',
                        'metadata': {
                            'themes': themes,
                            'spans': [],
                            'domain': 'News',
                            'source': 'DONALD'
                        }
                    }
            except Exception as e:
                print(f"[DONALD] error on {file.name}: {e}")