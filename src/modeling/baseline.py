from pathlib import Path
import pandas as pd
from loguru import logger
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

app = typer.Typer()

# mapping between dataset names and their corresponding source values in the metadata
SOURCE_MAP = {
    "ccn": "XAI-DisInfodemics",
    "voat": "Voat",
    # "semeval": "SemEval",
    "loco": "LOCO",
    "coco": "COCO",
    "donald": "DONALD"
}

@app.command()
def train(dataset_experiment: str = typer.Argument(..., help="dataset to train on (ccn, voat, semeval) or 'loo' for leave-one-out strategy"), loo: bool = typer.Option(False, "--loo", help="trains on the entire file, ignoring the filters")):
    # training and testing of a baseline in and cross domain
    logger.info(f"starting baseline. Traing dataset: {dataset_experiment.upper()}")
    # defining paths for training and testing data, as well as output directory
    train_path = Path("data/03_benchmark/benchmark_train.jsonl")
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
    except FileNotFoundError:
        logger.error(f"file not found: {train_path}")
        raise typer.Exit(code=1)
    if loo:
        logger.info(f"training on all data for strategy: leave-one-out")
        df_train = df_all_train
    # filtering the training data to only include rows corresponding to the specified dataset
    else:
        target_source = SOURCE_MAP.get(dataset_experiment.lower())
        df_train = df_all_train[df_all_train['metadata'].apply(lambda x: isinstance(x, dict) and x.get('source') == target_source)]
    if df_train.empty:
        logger.error(f"no training data found for experiment: {dataset_experiment}")
        raise typer.Exit(code=1)
    # isolating the text and labels from the data
    X_train_text= df_train["text"]
    y_train_label = df_train["canonical_label"]

    # random baseline: classifies texts randomly, while maintaining the actual class weights in the training dataset. 
    # used to demonstrate that the model’s results are not only due to class imbalance.
    logger.info("calculating Random Baseline...")
    dummy_model = DummyClassifier(strategy="stratified", random_state=42)
    dummy_model.fit(X_train_text, y_train_label)
    # training TF-IDF
    logger.info("text feature extraction (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000) 
    X_train_vectors = vectorizer.fit_transform(X_train_text)
    # training SVM
    logger.info("training Support Vector Machine...")
    model = SVC(kernel="linear", random_state=42) 
    model.fit(X_train_vectors, y_train_label)

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
        # applying the vocabulary learned during the training session
        X_test_vectors = vectorizer.transform(X_test_text)

        # SVM
        # making the predictions
        predictions_svm = model.predict(X_test_vectors)
        # saving the results memorizing test and training datasets
        output_file_svm = f"pred_baseline_train_{dataset_experiment}_test_{test_dataset}.jsonl"
        output_path_svm = output_dir / output_file_svm
        # saving id, label and predicted_label
        df_test["predicted_label"] = predictions_svm
        df_test[["original_id", "canonical_label", "predicted_label"]].to_json(output_path_svm, orient="records", lines=True)
        logger.success(f"   saved SVM: {output_path_svm}")

        # RANDOM BASELINE
        dummy_predictions = dummy_model.predict(X_test_text)
        file_dummy = f"pred_random_train_{dataset_experiment}_test_{test_dataset}.jsonl"
        output_path_dummy = output_dir / file_dummy
        df_test["predicted_label"] = dummy_predictions
        df_test[["original_id", "canonical_label", "predicted_label"]].to_json(output_path_dummy, orient="records", lines=True)
        logger.success(f"   saved DUMMY: {output_path_dummy}")

    logger.success(f"completed on training datasets: {dataset_experiment.upper()}")

if __name__ == "__main__":
    app()