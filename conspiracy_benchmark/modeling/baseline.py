from pathlib import Path
import yaml
import pandas as pd
from loguru import logger
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

app = typer.Typer()

@app.command()
def train(dataset: str):
    # training and testing of a baseline in and cross domain
    logger.info(f"starting baseline. Traing dataset: {dataset.upper()}")
    # reading paths dynamically
    config_path = Path("experiment-config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # creating paths to read and save the data
    input_dir = Path(config["directories"]["input_dir"])
    output_dir = Path(config["directories"]["output_dir"])
    # all datasets
    datasets = list(config["datasets"].keys())
    # check to see if the input dataset is in the list
    if dataset not in datasets:
        logger.error(f"Dataset {dataset} not found in experiment-config.yaml")
        raise typer.Exit(code=1)
    # loading training data
    train_splits = config["datasets"][dataset]["splits"]["train"]
    num_splits = len(train_splits)
    for i in range(num_splits):
        logger.info(f"processing split {i + 1} of {num_splits}")
        # loading training data by retrieving the exact string from the parallel array
        train_path = Path(train_splits[i])
        logger.info(f"loading training data from {train_path}...")
        # if not found continue to the next split
        try:
            df_train = pd.read_json(train_path, lines=True)
        except FileNotFoundError:
            logger.error(f"file not found: {train_path}")
            continue
    # isolating the text and labels from the data
    X_train_text= df_train["text"]
    y_train_label = df_train["label"]
    
    # random baseline: classifies texts randomly, while maintaining the actual class weights in the training dataset. 
    # used to demonstrate that the model’s results are not only due to class imbalance.
    logger.info("Calculating Random Baseline...")
    dummy_model = DummyClassifier(strategy="stratified", random_state=42)
    dummy_model.fit(X_train_text, y_train_label)

    # training TF-IDF and SVM
    logger.info("text feature extraction (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000) 
    X_train_vectors = vectorizer.fit_transform(X_train_text)

    logger.info("training Support Vector Machine...")
    model = SVC(kernel="linear", random_state=42) 
    model.fit(X_train_vectors, y_train_label)

    # testing in and cross domain
    logger.info("\nin-domain and cross-domain predictions...")
    for test_dataset in datasets:
        test_path = Path(config["datasets"][test_dataset]["splits"]["test"][i])
        if not test_path.exists():
            logger.warning(f"missing text file for {test_dataset}")
            continue
        logger.info(f"predicting on test set of: {test_dataset.upper()}")
        # current set
        df_test = pd.read_json(test_path, lines=True)
        X_test_text = df_test["text"]
        # applying the vocabulary learned during the training session
        X_test_vectors = vectorizer.transform(X_test_text)
        # new folder based on the split
        split_folder = str(i + 1)

        # SVM
        # making the predictions
        predictions_svm = model.predict(X_test_vectors)
        # saving the results memorizing test and training datasets
        output_file_svm = f"pred_baseline_train_{dataset}.jsonl"
        output_path_svm = output_dir / test_dataset / "splits" / split_folder / output_file_svm
        output_path_svm.parent.mkdir(parents=True, exist_ok=True)
        # saving id, label and predicted_label
        df_test["predicted_label"] = predictions_svm
        df_test[["id", "label", "predicted_label"]].to_json(output_path_svm, orient="records", lines=True)
        logger.success(f"   saved SVM: {output_path_svm}")

        # RANDOM BASELINE
        dummy_predictions = dummy_model.predict(X_test_text)
        file_dummy = f"pred_random_train_{dataset}.jsonl"
        output_path_dummy = output_dir / test_dataset / "splits" / split_folder / file_dummy
        df_test["predicted_label"] = dummy_predictions
        df_test[["id", "label", "predicted_label"]].to_json(output_path_dummy, orient="records", lines=True)
        logger.success(f"   saved DUMMY: {output_path_dummy}")

    logger.success(f"completed on training datasets: {dataset.upper()}")

if __name__ == "__main__":
    app()