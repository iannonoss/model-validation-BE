from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.base import clone
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import logging
import traceback
import io
import base64
import matplotlib.pyplot as plt
import math
import json
import re
import google.generativeai as genai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, mode=0o777, exist_ok=True)
    os.chmod(UPLOAD_DIR, 0o777)


# Schemi di richiesta
class PreprocessRequest(BaseModel):
    target: str
    features: list
    test_size: float = 0.2
    random_state: int = 0
    split_type: str = "train_test"


class TransformRequest(BaseModel):
    pass


# Variabili globali (per prototipazione)
global_df = None
X_train, X_test, y_train, y_test = None, None, None, None
categorical_cols, numerical_cols = [], []
scaler = None


# Funzione di pulizia per la serializzazione JSON
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        else:
            return obj
    else:
        return obj


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global global_df, categorical_cols, numerical_cols
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())
        df = pd.read_csv(file_location, on_bad_lines='skip')
        global_df = df

        columns = df.columns.tolist()
        missing_values = {col: int(val) for col, val in df.isnull().sum().items()}
        categorical_cols = [c for c in df.columns if df[c].dtype == object]
        numerical_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]

        preview_data = df.to_dict(orient="records")
        response_data = {
            "filename": file.filename,
            "columns": columns,
            "size": os.path.getsize(file_location),
            "missing_values": missing_values,
            "categorical_cols": categorical_cols,
            "numerical_cols": numerical_cols,
            "preview_data": preview_data
        }
        return clean_for_json(response_data)
    except Exception as e:
        logging.error("Errore nel caricamento del CSV: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def suggest_features_from_csv(df: pd.DataFrame) -> str:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel("gemini-1.5-flash")

    preview = df.head(20).to_csv(index=False)

    prompt = f"""
    Ti fornisco le prime 20 righe di un dataset in formato CSV. 
    Il tuo compito è suggerire:
    1. Quale colonna usare come target (obiettivo) per un modello di classificazione.
    2. Quali colonne usare come variabili indipendenti.
    3. Indica se ci sono colonne da escludere e perché.

    Dataset:
    {preview}

    Rispondi in formato JSON con le seguenti chiavi:
    {{
        "target": "...",
        "independent_features": ["...", "..."],
        "excluded_columns": ["..."],
        "motivation": "..."
    }}
    """

    response = model.generate_content(prompt)
    return response.text


@app.get("/suggest_features")
async def suggest_features():
    global global_df
    if global_df is None:
        raise HTTPException(status_code=400, detail="Nessun dataset caricato.")

    try:
        suggestion = suggest_features_from_csv(global_df)
        cleaned_text = re.sub(r"```json|```", "", suggestion).strip()
        suggestion_dict = json.loads(cleaned_text)
        return {"suggestion": suggestion_dict}
    except Exception as e:
        logging.error("Errore in suggest_features: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess_data")
async def preprocess_data(request: PreprocessRequest):
    global global_df, X_train, X_test, y_train, y_test, categorical_cols, numerical_cols
    if global_df is None:
        raise HTTPException(status_code=400, detail="Nessun dataset caricato.")
    df = global_df.copy()
    try:
        # Controlla la presenza del target e delle feature
        if request.target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target '{request.target}' non trovato.")
        for feat in request.features:
            if feat not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature '{feat}' non trovata.")

        X = df[request.features]
        y = df[request.target]
        if y.nunique() < 2:
            raise HTTPException(status_code=400, detail="Il target ha un solo valore.")
        if X.isnull().all().all():
            raise HTTPException(status_code=400, detail="Tutte le feature sono NaN.")

        # Split dei dati
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )
        numerical_cols = [c for c in X_train.columns if X_train[c].dtype in ["int64", "float64"]]
        categorical_cols = [c for c in X_train.columns if X_train[c].dtype == object]

        logging.info("Preprocessamento iniziale completato.")
        response_data = {
            "split_type": request.split_type,
            "test_size": request.test_size,
            "train_preview": X_train.to_dict(orient="records"),
            "valid_preview": X_test.to_dict(orient="records"),
        }
        return clean_for_json(response_data)
    except Exception as e:
        logging.error("Errore in preprocess_data: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transform_data")
async def transform_data(_: TransformRequest):
    global X_train, X_test, categorical_cols, numerical_cols, scaler
    if X_train is None or X_test is None:
        raise HTTPException(status_code=400, detail="Devi prima chiamare /preprocess_data.")
    try:
        # Imputazione dei valori mancanti
        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        if numerical_cols:
            X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
        if categorical_cols:
            X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        # Filtra solo le colonne categoriali con più di un valore unico
        categorical_cols = [c for c in categorical_cols if X_train[c].nunique() > 1]

        # Encoding one-hot per le colonne categoriali
        if categorical_cols:
            logging.info("🔄 Encoding delle variabili categoriali...")
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_train_enc = encoder.fit_transform(X_train[categorical_cols])
            X_test_enc = encoder.transform(X_test[categorical_cols])
            logging.info("Encoder categories: %s", encoder.categories_)
            logging.info("Shape X_train_enc: %s", X_train_enc.shape)
            feature_names = encoder.get_feature_names_out()
            if X_train_enc.shape[1] != len(feature_names):
                feature_names = feature_names[:X_train_enc.shape[1]]
            X_train_enc = pd.DataFrame(X_train_enc, columns=feature_names)
            X_test_enc = pd.DataFrame(X_test_enc, columns=feature_names)
            X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
            X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)
            X_train = pd.concat([X_train, X_train_enc], axis=1)
            X_test = pd.concat([X_test, X_test_enc], axis=1)

        # Normalizzazione delle feature numeriche originali
        if numerical_cols:
            scaler = StandardScaler()
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        logging.info("Trasformazione completata!")
        response_data = {
            "train_preview": X_train.to_dict(orient="records"),
            "valid_preview": X_test.to_dict(orient="records"),
        }
        return clean_for_json(response_data)
    except Exception as e:
        logging.error("Errore in transform_data: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Funzione per creare scatter plot in 2D (dati già ridotti)
def create_scatter_plot_2d(X_2d, y, clf, title):
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm_r")
    colors = ['red' if label == 0 else 'blue' for label in y]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, edgecolors="k", label="Dati")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Classe 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Classe 1')
    ]
    ax.legend(handles=legend_elements, title="Legenda")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_plot(X_orig, y, clf, title):
    x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
    y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_scaled = scaler.transform(mesh_points)
    additional_cols = X_train.shape[1] - 2
    if additional_cols > 0:
        mesh_scaled = np.hstack([mesh_scaled, np.zeros((mesh_scaled.shape[0], additional_cols))])
    Z = clf.predict(mesh_scaled)
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm_r")
    colors = ['red' if label == 0 else 'blue' for label in y]
    ax.scatter(X_orig[:, 0], X_orig[:, 1], c=colors, edgecolors="k", label="Dati")
    ax.set_xlabel("Feature 1 (Original Scale)")
    ax.set_ylabel("Feature 2 (Original Scale)")
    ax.set_title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@app.get("/predict")
async def predict_test_data(model_selected: str):
    global scaler, X_train, X_test, y_train, y_test, numerical_cols

    if model_selected == "Random Forest":
        clf = RandomForestClassifier(max_depth=5, random_state=0)
    elif model_selected == "SVM":
        clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    elif model_selected in ["XGBoost", "Neural Network"]:
        return {"message": "Work in progress"}
    else:
        return {"message": "Nessun modello scelto"}

    try:
        # Gestione Label Encoding per target categorico
        if not np.issubdtype(y_train.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        # Fit modello
        clf.fit(X_train, y_train)

        # Predizioni
        y_pred_test = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calcolo delle metriche di validazione
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1_score': f1_score(y_test, y_pred_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Creazione grafici decision boundary
        if len(numerical_cols) == 2:
            X_train_plot = X_train[numerical_cols].to_numpy()
            X_train_original = scaler.inverse_transform(X_train_plot)

            img_test = create_plot(scaler.inverse_transform(X_test[numerical_cols].to_numpy()),
                                   y_test, clf, f"Decision Boundary Test - {model_selected}")
        else:
            pca = PCA(n_components=2)
            X_train_2d = pca.fit_transform(X_train)
            X_test_2d = pca.transform(X_test)

            clf_2d = clone(clf)
            clf_2d.fit(X_train_2d, y_train)

            img_test = create_scatter_plot_2d(X_test_2d, y_test, clf_2d,
                                              f"Decision Boundary Test (PCA) - {model_selected}")

        response_data = {
            "model_selected": model_selected,
            "validation_metrics": metrics,
            "graph_test": f"data:image/png;base64,{img_test}"
        }

        return clean_for_json(response_data)
    except Exception as e:
        logging.error("Errore in predict_test_data: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
