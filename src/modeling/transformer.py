from pathlib import Path
import pandas as pd
from loguru import logger
import typer
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

app = typer.Typer()

# mapping between dataset names and their corresponding source values in the metadata
SOURCE_MAP = {
    "ccn": "XAI-DisInfodemics",
    "voat": "Voat",
    "semeval": "SemEval",
    "loco": "LOCO",
    "coco": "COCO",
    "donald": "DONALD"
}

# mapping between textual labels and numerical IDs for the Transformer model
LABEL_TO_ID = {"no_conspiracy": 0, "conspiracy": 1, "cant_tell": 2}
ID_TO_LABEL = {0: "no_conspiracy", 1: "conspiracy", 2: "cant_tell"}

@app.command()
def train(dataset_experiment: str = typer.Argument(..., help="dataset to train on (ccn, voat, semeval) or 'loo' for leave-one-out strategy"), loo: bool = typer.Option(False, "--loo", help="trains on the entire file, ignoring the filters")):
    # training and testing of a CT-BERT in and cross domain
    logger.info(f"starting CT-BERT. Traing dataset: {dataset_experiment.upper()}")
    # defining paths for training and testing data, as well as output directory
    train_path = Path("data/03_benchmark/benchmark_train.jsonl")
    val_path = Path("data/03_benchmark/benchmark_val.jsonl")
    test_path = Path("data/03_benchmark/benchmark_test.jsonl")
    output_dir = Path("reports/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    # all datasets
    datasets = list(SOURCE_MAP.keys())
    # check to see if the input dataset is in the list
    if not loo and dataset_experiment not in datasets:
        logger.error(f"dataset {dataset_experiment} not found")
        raise typer.Exit(code=1)
    # loading training data
    logger.info(f"loading training data from {train_path}...")
    try:
        df_all_train = pd.read_json(train_path, lines=True)
        df_all_val = pd.read_json(val_path, lines=True)
    except FileNotFoundError:
        logger.error(f"file not found: {train_path}")
        raise typer.Exit(code=1)
    if loo:
        logger.info(f"training on all data for strategy: leave-one-out")
        df_train = df_all_train
        df_val = df_all_val
    # filtering the training data to only include rows corresponding to the specified dataset
    else:
        target_source = SOURCE_MAP.get(dataset_experiment.lower())
        df_train = df_all_train[df_all_train['metadata'].apply(lambda x: isinstance(x, dict) and x.get('source') == target_source)]
        df_val = df_all_val[df_all_val['metadata'].apply(lambda x: isinstance(x, dict) and x.get('source') == target_source)]
    if df_train.empty:
        logger.error(f"no training data found for experiment: {dataset_experiment}")
        raise typer.Exit(code=1)
    # isolating the text and labels from the data
    X_train_text = df_train["text"]
    # mapping the textual labels to numerical IDs for the Transformer model
    y_train_label_ids = df_train["canonical_label"].map(LABEL_TO_ID).tolist()
    # loading validation data
    X_val_text = df_val["text"]
    y_val_label_ids = df_val["canonical_label"].map(LABEL_TO_ID).tolist()

    # tokenization and feature extraction using CT-BERT tokenizer
    logger.info("CT-BERT Tokenizer: text feature extraction...")
    model_name = "digitalepidemiologylab/covid-twitter-bert-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # creating a Hugging Face Dataset from the training data
    hf_train = Dataset.from_dict({"text": X_train_text.tolist(), "label": y_train_label_ids})
    hf_val = Dataset.from_dict({"text": X_val_text.tolist(), "label": y_val_label_ids})

    # tokenizes texts, padding or truncating them to a fixed length of 96 tokens , optimized for
    # short social media texts (as per CT-BERT paper), this ensures uniform tensor dimensions required for GPU batch processing.
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=96)
    # applying the tokenization function to the entire training dataset in a batched manner for efficiency
    hf_train_tokenized = hf_train.map(tokenize_function, batched=True)
    hf_val_tokenized = hf_val.map(tokenize_function, batched=True)

    # training CT-BERT
    logger.info("training CT-BERT Model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # training arguments for the Trainer
    training_args = TrainingArguments(
        output_dir="./models/ctbert_checkpoints",    # directory for model outputs
        num_train_epochs=3,                          # standard epochs for BERT fine-tuning
        per_device_train_batch_size=16,              # low batch size to prevent GPU out of memory errors
        per_device_eval_batch_size=64,               # higher batch size for faster predictions
        logging_steps=10,                            # print training loss every 10 steps
        eval_strategy="epoch",                       # evaluate the model at the end of each epoch  
        save_strategy="no"                           # disables checkpointing to save cluster disk space
    )
    # initializing the Trainer with the model, training arguments, and tokenized training dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_tokenized,
        eval_dataset=hf_val_tokenized,  
    )
    trainer.train()

    # testing in and cross domain
    logger.info("\nin-domain and cross-domain predictions...")
    # loading the test data
    try:
        df_all_test = pd.read_json(test_path, lines=True)
    except FileNotFoundError:
        logger.error(f"file not found: {test_path}")
        raise typer.Exit(code=1)
    # iterating over all datasets to perform predictions
    for test_dataset in datasets:
        logger.info(f"predicting on test set of: {test_dataset.upper()}")
        # current set
        # filtering the test data to only include rows corresponding to the current test dataset
        current_test_source = SOURCE_MAP.get(test_dataset.lower())
        df_test = df_all_test[df_all_test['metadata'].apply(lambda x: isinstance(x, dict) and x.get('source') == current_test_source)].copy()
        if df_test.empty:
            logger.warning(f"missing text file/data for {test_dataset}")
            continue
        X_test_text = df_test["text"]
        y_test_label_ids = df_test["canonical_label"].map(LABEL_TO_ID).tolist()
        # applying the vocabulary learned during the training session
        hf_test = Dataset.from_dict({"text": X_test_text.tolist(), "label": y_test_label_ids})
        hf_test_tokenized = hf_test.map(tokenize_function, batched=True)
        # making the predictions
        predictions_output = trainer.predict(hf_test_tokenized)
        predictions_ctbert_ids = np.argmax(predictions_output.predictions, axis=1)
        # mapping the predicted numerical IDs back to their corresponding textual labels
        predictions_ctbert_str = [ID_TO_LABEL[pred] for pred in predictions_ctbert_ids]
        # saving the results memorizing test and training datasets
        output_file_ctbert = f"pred_ctbert_train_{dataset_experiment}_test_{test_dataset}.jsonl"
        output_path_ctbert = output_dir / output_file_ctbert
        # saving id, label and predicted_label
        df_test["predicted_label"] = predictions_ctbert_str
        df_test[["original_id", "canonical_label", "predicted_label"]].to_json(output_path_ctbert, orient="records", lines=True)
        logger.success(f"   saved CT-BERT: {output_path_ctbert}")
    logger.success(f"completed on training datasets: {dataset_experiment.upper()}")

if __name__ == "__main__":
    app()