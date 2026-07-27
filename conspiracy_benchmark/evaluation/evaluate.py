from pathlib import Path
import yaml
import pandas as pd
from loguru import logger
import typer
from sklearn.metrics import classification_report, accuracy_score

app = typer.Typer()

@app.command()
def matrix(model_name: str):
    # evaluation of predictions made by the models
    logger.info(f"starting evaluation for model: {model_name.upper()}")
    # reading paths dynamically
    config_path = Path("experiment-config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    output_dir = Path(config["directories"]["output_dir"])
    # extracting datasets list from yaml file
    datasets = list(config["datasets"].keys())
    # lists to save the results
    visual_matrix_data = []
    detailed_metrics = []
    # creating the upper triangular matrix w/ train splits as rows, test splits as columns for all datasets, 
    # the diagonal is the in-domain performance, the off-diagonal is the cross-domain performance
    for i, train_ds in enumerate(datasets):
        for j, test_ds in enumerate(datasets):
            # skipping if the index of the column (test) is less than the index of the row (train) 
            # in this way a upper trinagular matrix is bulit with the diagonal as the in-domain performance
            if j < i:
                continue
            # number of splits
            # for now only one split, could be necessary to add more
            splits = len(config["datasets"][test_ds]["splits"]["test"])
            split = "1"
            # constructing file path to obtain the model's perfomance
            file_name = f"pred_{model_name}_train_{train_ds}.jsonl"
            file_path = output_dir / test_ds / "splits" / split / file_name
            if not file_path.exists():
                logger.warning(f"missing file: {file_path}")
                continue
            # reading the results
            df = pd.read_json(file_path, lines=True)
            y_true = df["label"]
            y_pred = df["predicted_label"]
            
            # calculating all the metrics (F1, acc, prec, rec per class)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            # calculating macro and weighted F1 for the matrix visualisation
            macro_f1 = report["macro avg"]["f1-score"]
            weighted_f1 = report["weighted avg"]["f1-score"]
            # saving macro f1 score for the visuakl matrix
            visual_matrix_data.append({
                "Train": train_ds.upper(),
                "Test": test_ds.upper(),
                "Macro_F1": round(macro_f1, 3)
            })
            # saving all the metrics used for the evaluation, separating the metrics for better reading of the results
            detailed_metrics.append({
                "train": train_ds.upper(),
                "test": test_ds.upper(),
                "accuracy": round(acc, 3),
                "macro_f1": round(macro_f1, 3),
                "weighted_f1": round(weighted_f1, 3),
                "class_metrics": {k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]}
            })

    # rendering the matrix in a 2D table for printing on the terminal
    df_matrix = pd.DataFrame(visual_matrix_data)
    pivot_matrix = df_matrix.pivot(index="Train", columns="Test", values="Macro_F1")
    # reordering the rows and columns to match the original order (pandas sorts alphabetically by default)
    labels_upper = [d.upper() for d in datasets]
    pivot_matrix = pivot_matrix.reindex(index=labels_upper, columns=labels_upper)
    # printing the matrix
    print(f"\nUPPER TRIANGULAR MATRIX (Macro F1), model: {model_name.upper()}")
    # filling voids with "-" to preserve the look of the matrix 
    print(pivot_matrix.fillna("-").to_string())

    # saving the results and the matrix
    # saving the matrix as csv
    matrix_file = output_dir / f"matrix_{model_name}.csv"
    # filling the voids with "-" to preserve the triangular look and manually inserting decimals to avoid it being read as text
    matrix_csv = pivot_matrix.copy()
    for col in matrix_csv.columns:
        matrix_csv[col] = matrix_csv[col].apply(lambda x: f"{x:.3f}".replace('.', ',') if pd.notna(x) else "-")
    matrix_csv.to_csv(matrix_file, sep=";")
    logger.success(f"triangular matrix saved to: {matrix_file}")
    # saving the report as json
    report_file = output_dir / f"evaluation_report_{model_name}.json"
    pd.DataFrame(detailed_metrics).to_json(report_file, orient="records", indent=4)
    logger.success(f"full detailed metrics saved to: {report_file}")

if __name__ == "__main__":
    app()