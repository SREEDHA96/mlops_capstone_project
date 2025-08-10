# src/model/model_evaluation.py

import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn
import dagshub

# use your project logger (remove stdlib logging import)
from src.logger import logging


# ---------- MLflow tracking (DagsHub) ----------
# Use ONE setup. dagshub.init sets the tracking URI for you.
dagshub.init(repo_owner="SREEDHA96", repo_name="mlops_capstone_project", mlflow=True)
# ------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logging.error("Model file not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error loading model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logging.error("Error loading data from %s: %s", file_path, e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_pred_proba)),
        }
        logging.info("Computed evaluation metrics")
        return metrics
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise


def save_json(payload: dict, file_path: str) -> None:
    """Save a dictionary to JSON."""
    try:
        with open(file_path, "w") as f:
            json.dump(payload, f, indent=4)
        logging.info("Saved JSON to %s", file_path)
    except Exception as e:
        logging.error("Error saving JSON to %s: %s", file_path, e)
        raise


def main():
    # Make sure relative paths behave as expected
    logging.info("CWD: %s", os.getcwd())

    # Always have a place to write reports
    os.makedirs("reports", exist_ok=True)

    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model_artifact_path = "model"  # folder name under MLflow artifacts

        try:
            # ---- Load model & data ----
            clf = load_model("./models/model.pkl")
            test_df = load_data("./data/processed/test_bow.csv")

            X_test = test_df.iloc[:, :-1].to_numpy()
            y_test = test_df.iloc[:, -1].to_numpy()

            # ---- Evaluate ----
            metrics = evaluate_model(clf, X_test, y_test)

            # ---- Write experiment_info.json EARLY so it always exists ----
            experiment_info_path = "reports/experiment_info.json"
            save_json({"run_id": run_id, "model_path": model_artifact_path}, experiment_info_path)

            # ---- Persist metrics locally ----
            metrics_path = "reports/metrics.json"
            save_json(metrics, metrics_path)

            # ---- Log to MLflow ----
            # metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # params (if available)
            if hasattr(clf, "get_params"):
                for k, v in clf.get_params().items():
                    # Ensure JSON-serializable
                    mlflow.log_param(k, str(v))

            # model artifact
            mlflow.sklearn.log_model(clf, model_artifact_path)

            # report files as artifacts
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(experiment_info_path)

            # helpful tags for later lookup (no registry)
            mlflow.set_tag("candidate_model_uri", f"runs:/{run_id}/{model_artifact_path}")

            logging.info("Evaluation complete. Run ID: %s", run_id)

        except Exception as e:
            logging.error("Failed to complete the model evaluation process: %s", e)
            print(f"Error: {e}")
            # experiment_info.json was already written above; leaves breadcrumbs even on failure


if __name__ == "__main__":
    main()
