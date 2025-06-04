# 🌸 MLOps Project - Iris Classification

Ce projet est une démonstration complète d’un pipeline **MLOps** avec le dataset classique **Iris**. Il combine entraînement de modèles, suivi avec MLflow, API FastAPI pour l'entraînement et la prédiction, interface Streamlit, PostgreSQL comme backend MLflow, et Docker/Docker Compose pour l’orchestration.

---

## 📌 Objectifs

- Entraîner des modèles `RandomForest`, `LogisticRegression` ou `KNN` sur le dataset Iris
- Suivre les expériences avec MLflow (params, métriques, artefacts)
- Exposer une API avec FastAPI pour entraîner ou prédire
- Fournir une interface utilisateur avec Streamlit
- Conteneuriser le tout avec Docker Compose

---

## 🧱 Structure du projet

├── app/
│ ├── model/                # Contient train.py et modèle enregistré
│ │ └── train.py
│ ├── streamlit_app.py      # Interface Streamlit
│ └── api_app.py            # API FastAPI (train + predict)
├── data/                   # Dossier monté pour partage entre services
├── docker/
│ └── Dockerfile            # Unique Dockerfile pour tous les services Python
├── docker-compose.yml
├── mlruns/                 # Artéfacts MLflow (param, metrics, model)
├── confusion_matrix.png    # Image sauvegardée par MLflow
└── README.md

## Lancer tous les services
    bash
    docker-compose up --build
Cela démarre :

PostgreSQL (pour MLflow)

- MLflow Tracking Server sur http://localhost:5000

- FastAPI sur http://localhost:8000

- Streamlit sur http://localhost:8501

- pgAdmin sur http://localhost:8080

- Portainer (gestion Docker) sur http://localhost:9000

## 🚀 Utilisation

# 🔧 Entraîner un modèle (depuis Streamlit)

Aller sur http://localhost:8501

Choisir le modèle et les hyperparamètres

Lancer l'entraînement

Voir les métriques sur MLflow UI

# 🔮 Faire une prédiction
Toujours via Streamlit, dans l’onglet Prédiction, entrer les caractéristiques (sépal/pétale), et obtenir la classe prédite (Setosa, Versicolor, Virginica).

## 📦 API FastAPI

POST /train — Lancer un entraînement

POST /predict — Faire une prédiction simple

## 🧪 Tech Stack

Python 3.11

scikit-learn

FastAPI

Streamlit

MLflow

Docker & Docker Compose

PostgreSQL

pgAdmin & Portainer

📌 Auteur
👤 Tarik ZOUBIR — Projet démonstration MLOps pour Iris Dataset



