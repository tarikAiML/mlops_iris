# Tarik ZOUBIR 
# For testing full MLOPS process
import argparse
#import joblib
import mlflow
import os
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# For parsing args
def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier on the Iris dataset")
    parser.add_argument("--model", required=True, type=str, default="random_forest",
                        choices=["random_forest", "logistic_regression", "knn"],
                        help="Model to train")
    parser.add_argument("--n_estimators", type=int, default=100, help="For Random Forest")
    parser.add_argument("--n_neighbors", type=int, default=5, help="For KNN")
    return parser.parse_args()

# Function for loading data from sklearn.datasets
def load_data():
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Function for training model
def train_model(args, X_train, y_train):
    if args.model == "random_forest":
        model = RandomForestClassifier(n_estimators=args.n_estimators)
    elif args.model == "logistic_regression":
        model = LogisticRegression(max_iter=200)
    elif args.model == "knn":
        model = KNeighborsClassifier(n_neighbors=args.n_neighbors)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    model.fit(X_train, y_train)
    return model

# Function for evaluating model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="weighted"),
        "Recall": recall_score(y_test, preds, average="weighted"),
        "F1": f1_score(y_test, preds, average="weighted"),
    }
    return metrics

# The main function
def main():
    # ---------- 1. ARGS ----------
    args = parse_args()
    
    # ---------- 2. MLflow ----------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"mlops_iris_{args.model}")
    
    # ---------- 3. Data ----------
    X_train, X_test, y_train, y_test = load_data()
    
    # ---------- 4. Training ----------
    with mlflow.start_run():
        model = train_model(args, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Log params & metrics
        mlflow.log_param("model", args.model)
        if args.model == "random_forest":
            mlflow.log_param("n_estimators", args.n_estimators)
        elif args.model == "knn":
            mlflow.log_param("n_neighbors", args.n_neighbors)

        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        print(f"Training done: {args.model}, metrics: {metrics}")

        # Save local model for API
        #joblib.dump(model, "app/model.joblib")
        #print(f"Model saved and metrics: {metrics}")
        
        # Log confusion matrix as artifact
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")


if __name__ == "__main__":
    main()
