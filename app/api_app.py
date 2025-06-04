from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, uuid, os
from typing import Optional
import numpy as np
import joblib
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="IRIS API")

# Instrumentation Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

class TrainRequest(BaseModel):
    model: str       # RandomForest | LogisticRegression | Knn
    n_estimators: int  | None = None   # ignoré hors RandomForest
    n_neighbors: int | None = None   # ignoré hors KNN

@app.post("/train")
def train(req: TrainRequest):
    """Lance un entraînement MLflow en sous‑processus."""
    run_id = str(uuid.uuid4())[:8]
    cmd = [
        "python", "train.py",
        "--model", req.model
    ]
    if req.model == "random_forest":
        cmd += ["--n_estimators", str(req.n_estimators)]
    if req.model == "knn":
        cmd += ["--n_neighbors", str(req.n_neighbors)]
    # On passe la variable d'env pour que le sous‑processus parle à MLflow
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    subprocess.Popen(cmd, env=env)
    return {"status": "started", "run_id": run_id, "cmd": " ".join(cmd)}

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(inp: PredictRequest):
    # Exemple de chargement d'un modèle simple (joblib, pas MLflow ici pour l'instant)
    model_path = "app/model.joblib"
    if not os.path.exists(model_path):
        return {"error": "Modèle non entraîné encore. Lancez d'abord /train."}

    model = joblib.load(model_path)

    input_array = np.array([[inp.sepal_length, inp.sepal_width, inp.petal_length, inp.petal_width]])
    prediction = model.predict(input_array)[0]

    # Classes names
    class_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction": int(prediction),
        "species": class_names[prediction]
    }