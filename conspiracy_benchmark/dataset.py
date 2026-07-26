from pathlib import Path
import yaml
from loguru import logger
import typer
import pandas as pd
import json
from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence

app = typer.Typer()

# general features definition
benchmark_features = Features({
    'id': Value('string'),
    'text': Value('string'),
    # ClassLabel to map non-conspiracy, conspiracy and uncertain texts
    'label': ClassLabel(num_classes=3, names=['no_conspiracy', 'conspiracy', 'cant_tell']),
    # lists of themes in the text
    'themes': Sequence(Value('string')),
    # lists of markers in the text and the associated label
    'spans': [{                  
        'start_char': Value('int32'),
        'end_char': Value('int32'),
        'label': Value('string'),
        'span_text': Value('string')
    }],
    # social or news
    'domain': Value('string'),
    # original dataset
    'source': Value('string')
})

# values mapped for each dataset: 0 non-conspiracy and 1 conspiracy
LABEL_MAP = {
    True: 1, False: 0,
    'TRUE': 1, 'FALSE': 0,
    'True': 1, 'False': 0,
    1: 1, 0: 0,
    'CONSPIRACY': 1, 'CRITICAL': 0,
    'UNRELATED': 0,
    "Can't tell": 2, 
    "yes": 1, "no": 0,
    "Yes": 1, "No": 0,
    "YES": 1, "NO": 0,
    "conspiracy": 1, "mainstream": 0,
}

# Voat
def harmonize_voat(example):
    # themes
    column_themes = ['Actor', 'Action', 'Threat', 'Pattern', 'Secrecy']
    active_themes = []
    spans = []
    text =  example.get('body', "")
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
    doc_id = example.get('comment_id') if example.get('comment_id') else example.get('submission_id')
    # original label
    raw_label = example.get('overall_ct', "Can't tell")
    # final harmonized result
    return {
        'id': str(doc_id),
        'text': text,
        'label': LABEL_MAP.get(raw_label, 0),
        'themes': active_themes,
        'spans': spans,
        'domain': 'Social Media',
        'source': 'Voat'
    }

# SemEval
def harmonize_semeval(example):
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
    # final harmonized result
    return {
        'id': str(example.get('_id', '')),
        'text': example.get('text', ''),
        'label': LABEL_MAP.get(example.get('conspiracy', "Can't tell"), 0),
        'themes': themes,
        'spans': spans, 
        'domain': 'Social Media',
        'source': 'SemEval'
    }

# CCN
def harmonize_ccn(example):
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
    # final harmonized result
    return {
        'id': str(example.get('id', '')),
        'text': example.get('text', ''),
        'label': LABEL_MAP.get(example.get('category', "Can't tell"), 0),
        'themes': themes,
        'spans': spans,
        'domain': 'Social Media',
        'source': 'XAI-DisInfodemics'
    }

@app.command()
def harmonize():
    # harmonizing the datasets and splitting them into train/val/test
    logger.info("start reading configuration...")
    # reading file yaml to obtain datasets
    config_path = Path("experiment-config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    import_dir = Path(config["directories"]["import_dir"])
    input_dir = Path(config["directories"]["input_dir"])
    
    # splitting of the datasets: 80% for training and 10% each for validation and testing
    def splits(dataset, dataset_name):
        logger.info(f"[{dataset_name}] split creation 80/10/10...")
        # 80% for testing
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        # remaining 20% split into two 10%
        val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
        splits = {
            "train": train_test['train'],
            "val": val_test['train'],
            "test": val_test['test']
        }
        # directory to save the splits
        save_dir = input_dir / dataset_name / "splits" / "1"
        save_dir.mkdir(parents=True, exist_ok=True)
        # saving the results as .jsonl
        for split_name, split_dataset in splits.items():
            file_path = save_dir / f"{split_name}.jsonl"
            split_dataset.to_json(file_path, orient="records", lines=True)
        logger.success(f"[{dataset_name}] saved in {save_dir}")

    # VOAT
    logger.info("processing Voat...")
    path_voat = import_dir / "voat" / "LLMs-and-conspiracy-beliefs-main" / "data" / "voat_annotation.csv"
    voat_raw = load_dataset("csv", data_files=str(path_voat), split="train")
    # apply harmonizer, drop old columns, and enforce unified schema
    ds_voat = voat_raw.map(harmonize_voat, remove_columns=voat_raw.column_names, features=benchmark_features)
    splits(ds_voat, "voat")

    # SEMEVAL
    logger.info("processing SemEval...")
    # building dynamic paths
    path_semeval_labels = import_dir / "semeval26_starter_pack" / "train_redacted.jsonl"
    path_semeval_texts = import_dir / "semeval26_starter_pack" / "train_rehydrated.jsonl"
    # merging ids and annotated labels
    df_labels = pd.read_json(path_semeval_labels, lines=True)
    df_texts = pd.read_json(path_semeval_texts, lines=True)
    df_semeval = pd.merge(df_labels, df_texts[['_id', 'text']], on="_id", how="inner")
     # convert the pandas dataframe into a hugging face dataset
    semeval_raw = Dataset.from_pandas(df_semeval)
    ds_semeval = semeval_raw.map(harmonize_semeval, remove_columns=semeval_raw.column_names, features=benchmark_features)
    # splitting the dataset and saving the result
    splits(ds_semeval, "semeval")

    # CCN (XAI-DisInfodemics Corpus)
    logger.info("processing CCN...")
    path_ccn = import_dir / "CCN" / "dataset_en_gold.json"
    ccn_raw = load_dataset("json", data_files=str(path_ccn), split="train")
    ds_ccn = ccn_raw.map(harmonize_ccn, remove_columns=ccn_raw.column_names, features=benchmark_features)
    
    splits(ds_ccn, "ccn")

    logger.success("the datasets were harmonized, splitted and saved successfully")

if __name__ == "__main__":
    app()