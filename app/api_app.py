from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import os, subprocess, uuid
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Optional

def load_latest_metrics(experiment_name="mlops_iris_random_forest"):
    client = mlflow.tracking.MlflowClient()
    exp_id = get_experiment_id(experiment_name)
    if not exp_id:
        return

    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        return

    metrics = runs[0].data.metrics
    if "accuracy" in metrics:
        accuracy_gauge.set(metrics["accuracy"])
    if "f1_score" in metrics:
        f1_score_gauge.set(metrics["f1_score"])

app = FastAPI(title="IRIS API")

# Prometheus
Instrumentator().instrument(app).expose(app)
# Charger les dernières métriques connues depuis MLflow
load_latest_metrics()


# Variables globales
_cached_model = None
# Prometheus metrics personnalisées
accuracy_gauge = Gauge("iris_model_accuracy", "Accuracy of the Iris model")
f1_score_gauge = Gauge("iris_model_f1_score", "F1 Score of the Iris model")



# ------------------- UTILS -------------------
class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class TrainRequest(BaseModel):
    model: str  # random_forest | logistic_regression | knn
    n_estimators: Optional[int] = None
    n_neighbors: Optional[int] = None


def get_experiment_id(name: str):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp else None


def get_latest_model_uri(experiment_name: str):
    exp_id = get_experiment_id(experiment_name)
    if not exp_id:
        return None
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        return None
    return f"runs:/{runs[0].info.run_id}/model"

def load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_iris_random_forest")
    model_uri = get_latest_model_uri(experiment_name)
    if model_uri:
        _cached_model = mlflow.sklearn.load_model(model_uri)
    return _cached_model


# ------------------- ENDPOINTS -------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Iris MLOps API 🚀"}


@app.get("/confusion_matrix")
def confusion_matrix():
    if os.path.exists("confusion_matrix.png"):
        return FileResponse("confusion_matrix.png", media_type="image/png")
    return {"error": "Fichier non trouvé. Lancez /train pour générer le modèle."}


@app.post("/train")
def train(req: TrainRequest):
    run_id = str(uuid.uuid4())[:8]
    cmd = ["python", "model/train.py", "--model", req.model]

    if req.model == "random_forest" and req.n_estimators:
        cmd += ["--n_estimators", str(req.n_estimators)]
    if req.model == "knn" and req.n_neighbors:
        cmd += ["--n_neighbors", str(req.n_neighbors)]

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    subprocess.Popen(cmd, env=env)

    return {"status": "training started", "cmd": " ".join(cmd), "run_id": run_id}


@app.post("/predict")
def predict(input_data: PredictRequest):
    model = load_model()
    if model is None:
        return {"error": "Aucun modèle chargé. Lancez d'abord /train."}

    data = np.array([[input_data.sepal_length, input_data.sepal_width,
                      input_data.petal_length, input_data.petal_width]])
    prediction = model.predict(data)[0]
    class_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction": int(prediction),
        "species": class_names[prediction]
    }
