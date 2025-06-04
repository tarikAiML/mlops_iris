import streamlit as st
import requests, json, pandas as pd
import os
import joblib
import numpy as np
from PIL import Image

API_URL = os.getenv("API_URL", "http://api:8000")  # dans Docker

st.set_page_config(page_title="Prédiction IRIS", page_icon="🌸")
st.title("🌸 Prédiction de fleur IRIS (MLOps Demo)")
st.markdown(
    "Lancez des entraînements MLflow et observez‑les en temps réel dans "
    "[MLflow UI](http://20.151.96.60:5000).")

# ---------- Onglets Streamlit ----------
tab_train, tab_predict = st.tabs(["🧠 Entraînement", "🔮 Prédiction"])

# ========== 🧠 Onglet Entraînement ==========
with tab_train:
    st.header("Entraîner un modèle")
    model_type = st.selectbox("Choisir un modèle :", ["random_forest", "logistic_regression", "knn"])

    n_estimators = None
    n_neighbors = None

    if model_type == "random_forest":
        n_estimators = st.slider("n_estimators", 10, 300, 100, step=10)
    if model_type == "knn":
        n_neighbors = st.slider("n_neighbors", 1, 20, 5)

    if st.button("🚀 Lancer l'entraînement"):
        data = {"model": model_type}
        if n_estimators: data["n_estimators"] = n_estimators
        if n_neighbors: data["n_neighbors"] = n_neighbors

        with st.spinner("Entraînement en cours..."):
            res = requests.post(f"{API_URL}/train", json=data)
            if res.status_code == 200:
                st.success(f"Modèle {model_type} lancé avec succès (run_id: {res.json()['run_id']})")
            else:
                st.error("Erreur lors de l'entraînement")

# ========== 🔮 Onglet Prédiction ==========
with tab_predict:
    st.header("Faire une prédiction")

    sepal_length = st.slider("Longueur sépale (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Largeur sépale (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Longueur pétale (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Largeur pétale (cm)", 0.1, 2.5, 0.2)

    if st.button("🔎 Prédire"):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        try:
            model = joblib.load("app/model.joblib")
            prediction = model.predict([features])[0]
            classes = ["Setosa", "Versicolor", "Virginica"]
            st.success(f"🌺 Prédit : **{classes[prediction]}** (classe {prediction})")
        except FileNotFoundError:
            st.error("Modèle introuvable. Lancez un entraînement d'abord.")

    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)