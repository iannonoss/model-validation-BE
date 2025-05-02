# Standard library
import base64
import io
import json
import logging
import math
import os
import re
import traceback

# Environment & configuration
from dotenv import load_dotenv

# Data manipulation
import numpy as np
import pandas as pd

# Plotting
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Web backend (FastAPI)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML Core - scikit-learn
from sklearn import svm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold, KFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Advanced ML
import xgboost as xgb
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

class CrossValidateRequest(BaseModel):
    model_selected: str = Field(..., description="Modello da usare (es. 'Random Forest', 'SVM')")
    validation_type: str = Field(..., description="Tipo di validazione (es. 'k-fold', 'stratified-k-fold', 'loocv')")
    folds: int = Field(5, description="Numero di fold per K-Fold e Stratified K-Fold (ignorato per LOOCV)")

class InstanceRequest(BaseModel):
    instance: dict

# Variabili globali
global_df = None
X_train, X_test, y_train, y_test = None, None, None, None
categorical_cols, numerical_cols = [], []
scaler = None
global_X = None
global_y = None
global_full_numerical_cols = []
global_full_categorical_cols = []
global_num_imputer = None
global_cat_imputer = None
global_encoder = None
global_scaler = None
global_feature_columns = None
clf_model = None
global_num_imputer = None
global_cat_imputer = None
global_encoder = None
transformed_feature_names = None
# Funzione per creare un grafico 2D con PCA e confini decisionali

def create_scatter_plot_2d(X_2d, y, clf, title):
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")

    unique_labels = np.unique(y)
    colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        points = X_2d[y == label]
        ax.scatter(points[:, 0], points[:, 1], color=colors[idx], label=f"Classe {label}", edgecolors="k")

    ax.legend(title="Legenda")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Funzione per creare un grafico decisionale bidimensionale su dati originali

def create_plot(X_orig, y, clf, title):
    x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
    y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    Z = clf.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm_r")

    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for idx, label in enumerate(unique_labels):
        points = X_orig[y == label]
        ax.scatter(points[:, 0], points[:, 1],
                   color=colors[idx % len(colors)],
                   label=f"Classe {label}",
                   edgecolors="k")

    ax.legend(title="Legenda")
    ax.set_xlabel("Feature 1 (Original Scale)")
    ax.set_ylabel("Feature 2 (Original Scale)")
    ax.set_title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Funzione ricorsiva per serializzare oggetti complessi in formato JSON

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# Funzione per suggerire target e feature da un DataFrame usando il modello generativo

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

# Funzione per gestire valori float non serializzabili

def safe_float(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)

# Funzione wrapper per ROC AUC multiclass con gestione errori

def multiclass_roc_auc_score(y_true, y_pred, average="macro"):
    try:
        return roc_auc_score(y_true, y_pred, multi_class='ovr', average=average)
    except ValueError:
        return float('nan')
# Endpoint per il caricamento di un file CSV e l'analisi preliminare delle colonne

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global global_df, categorical_cols, numerical_cols
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Salva il file nella directory di upload
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Carica il contenuto del CSV in un DataFrame
        df = pd.read_csv(file_location, on_bad_lines='skip')
        global_df = df

        # Estrazione delle colonne e informazioni sui valori mancanti
        columns = df.columns.tolist()
        missing_values = {col: int(val) for col, val in df.isnull().sum().items()}

        # Identificazione delle colonne categoriche e numeriche
        categorical_cols = [c for c in df.columns if df[c].dtype == object]
        numerical_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]

        # Anteprima del dataset
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

# Endpoint per il suggerimento automatico di target e feature

@app.get("/suggest_features")
async def suggest_features():
    global global_df
    if global_df is None:
        raise HTTPException(status_code=400, detail="Nessun dataset caricato.")

    try:
        # Generazione del suggerimento tramite modello generativo
        suggestion = suggest_features_from_csv(global_df)

        # Rimozione di eventuali delimitatori di codice
        cleaned_text = re.sub(r"```json|```", "", suggestion).strip()
        suggestion_dict = json.loads(cleaned_text)

        return {"suggestion": suggestion_dict}
    except Exception as e:
        logging.error("Errore in suggest_features: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per la selezione del target, delle feature e per la suddivisione train/test

@app.post("/preprocess_data")
async def preprocess_data(request: PreprocessRequest):
    global selected_independent_features
    selected_independent_features = request.features
    global global_df, X_train, X_test, y_train, y_test
    global categorical_cols, numerical_cols
    global global_X, global_y, global_full_numerical_cols, global_full_categorical_cols

    if global_df is None:
        raise HTTPException(status_code=400, detail="Nessun dataset caricato.")

    df = global_df.copy()
    try:
        # Verifica presenza del target e delle feature
        if request.target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target '{request.target}' non trovato.")
        for feat in request.features:
            if feat not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature '{feat}' non trovata.")

        # Estrazione delle feature indipendenti e del target
        X_full = df[request.features].copy()
        y_full = df[request.target].copy()

        # Verifica validità del target e delle feature
        if y_full.nunique() < 2:
            raise HTTPException(status_code=400, detail="Il target ha un solo valore.")
        if X_full.isnull().all().all():
            raise HTTPException(status_code=400, detail="Tutte le feature sono NaN.")

        # Codifica del target se non numerico
        if not np.issubdtype(y_full.dtype, np.number):
            logging.info(f"Target '{request.target}' è categorico. Applicazione LabelEncoder.")
            le = LabelEncoder()
            y_full_encoded = le.fit_transform(y_full)
        else:
            y_full_encoded = y_full

        # Assegnazione alle variabili globali per cross-validation
        global_X = X_full
        global_y = y_full_encoded

        # Identificazione delle colonne numeriche e categoriche nel dataset completo
        global_full_numerical_cols = [c for c in global_X.columns if global_X[c].dtype in ["int64", "float64"]]
        global_full_categorical_cols = [c for c in global_X.columns if global_X[c].dtype == object]
        logging.info(f"Colonne numeriche globali identificate: {global_full_numerical_cols}")
        logging.info(f"Colonne categoriche globali identificate: {global_full_categorical_cols}")

        # Suddivisione in training e test set
        X_train, X_test, y_train, y_test = train_test_split(
            global_X,
            global_y,
            test_size=request.test_size,
            random_state=request.random_state,
            stratify=global_y if request.split_type == 'stratified' else None
        )

        # Identificazione delle colonne numeriche e categoriche nei dati di train
        numerical_cols = [c for c in X_train.columns if X_train[c].dtype in ["int64", "float64"]]
        categorical_cols = [c for c in X_train.columns if X_train[c].dtype == object]

        logging.info("Selezione features/target e preparazione variabili globali completata.")
        logging.info("Split Train/Test eseguito.")

        # Costruzione della risposta con anteprima dello split
        response_data = {
            "split_type": request.split_type,
            "test_size": request.test_size,
            "train_preview": X_train.head().to_dict(orient="records"),
            "valid_preview": X_test.head().to_dict(orient="records"),
            "global_numerical_features": global_full_numerical_cols,
            "global_categorical_features": global_full_categorical_cols
        }
        return clean_for_json(response_data)
    except Exception as e:
        logging.error("Errore in preprocess_data: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per la trasformazione dei dati: imputazione, encoding e scaling

@app.post("/transform_data")
async def transform_data(_: TransformRequest):
    global X_train, X_test, categorical_cols, numerical_cols
    global global_num_imputer, global_cat_imputer, global_encoder, global_scaler
    global transformed_feature_names, global_feature_columns

    if X_train is None or X_test is None:
        raise HTTPException(status_code=400, detail="Devi prima chiamare /preprocess_data.")

    try:
        # Imputazione dei valori mancanti
        global_num_imputer = SimpleImputer(strategy="mean")
        global_cat_imputer = SimpleImputer(strategy="most_frequent")

        if numerical_cols:
            X_train[numerical_cols] = global_num_imputer.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = global_num_imputer.transform(X_test[numerical_cols])

        if categorical_cols:
            X_train[categorical_cols] = global_cat_imputer.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = global_cat_imputer.transform(X_test[categorical_cols])

        # Selezione delle colonne categoriche con almeno due classi distinte
        valid_categorical_cols = [c for c in categorical_cols if X_train[c].nunique() > 1]

        # Codifica one-hot delle variabili categoriche valide
        if valid_categorical_cols:
            logging.info("Esecuzione encoding delle variabili categoriche...")
            global_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_train_enc = global_encoder.fit_transform(X_train[valid_categorical_cols])
            X_test_enc = global_encoder.transform(X_test[valid_categorical_cols])

            feature_names = global_encoder.get_feature_names_out(valid_categorical_cols)
            transformed_feature_names = feature_names.tolist()

            X_train_enc_df = pd.DataFrame(X_train_enc, columns=feature_names, index=X_train.index)
            X_test_enc_df = pd.DataFrame(X_test_enc, columns=feature_names, index=X_test.index)

            X_train = X_train.drop(columns=valid_categorical_cols)
            X_test = X_test.drop(columns=valid_categorical_cols)
            X_train = pd.concat([X_train, X_train_enc_df], axis=1)
            X_test = pd.concat([X_test, X_test_enc_df], axis=1)

        # Scaling delle variabili numeriche
        if numerical_cols:
            global_scaler = StandardScaler()
            X_train[numerical_cols] = global_scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = global_scaler.transform(X_test[numerical_cols])

        logging.info("Trasformazione completata.")
        global_feature_columns = X_train.columns.tolist()

        # Costruzione della risposta con anteprima dei dati trasformati
        response_data = {
            "train_preview": X_train.to_dict(orient="records"),
            "valid_preview": X_test.to_dict(orient="records"),
            "transformed_columns": global_feature_columns
        }
        return clean_for_json(response_data)

    except Exception as e:
        logging.error("Errore in transform_data: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per l'addestramento del modello selezionato e la generazione dei grafici decisionali

@app.get("/predict")
async def predict_test_data(model_selected: str):
    global global_scaler, X_train, X_test, y_train, y_test, numerical_cols, clf_model

    # Selezione del modello in base al parametro fornito
    if model_selected == "Random Forest":
        clf = RandomForestClassifier(max_depth=5, random_state=0)
    elif model_selected == "SVM":
        clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    elif model_selected == "XGBoost":
        n_classes = len(np.unique(y_train))
        clf = xgb.XGBClassifier(
            objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
            num_class=n_classes if n_classes > 2 else None,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    elif model_selected == "Neural Network":
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
    else:
        return {"message": "Nessun modello scelto"}

    clf_model = clf

    try:
        # Codifica del target se necessario
        if not np.issubdtype(y_train.dtype, np.number):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        # Addestramento del modello
        clf.fit(X_train, y_train)

        y_pred_test = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        n_classes = y_pred_proba.shape[1]
        roc_auc = None

        # Calcolo della metrica ROC AUC
        try:
            if n_classes > 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        except ValueError as e:
            logging.warning(f"Impossibile calcolare ROC AUC: {str(e)}")
            roc_auc = None

        # Calcolo delle metriche di valutazione
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='macro'),
            'recall': recall_score(y_test, y_pred_test, average='macro'),
            'f1_score': f1_score(y_test, y_pred_test, average='macro'),
            'roc_auc': roc_auc
        }

        # Generazione dei grafici decisionali
        if len(numerical_cols) == 2:
            # Inversione dello scaling per le due feature numeriche
            X_train_plot = global_scaler.inverse_transform(X_train[numerical_cols])
            X_test_plot = global_scaler.inverse_transform(X_test[numerical_cols])

            # Addestramento del modello sui dati non scalati
            clf_plot = clone(clf)
            clf_plot.fit(X_train_plot, y_train)

            # Creazione dei grafici
            img_train = create_plot(X_train_plot, y_train, clf_plot, f"Decision Boundary Train - {model_selected}")
            img_test = create_plot(X_test_plot, y_test, clf_plot, f"Decision Boundary Test - {model_selected}")

        else:
            # Applicazione PCA se le feature sono più di 2
            pca = PCA(n_components=2)
            X_train_2d = pca.fit_transform(X_train)
            X_test_2d = pca.transform(X_test)

            clf_2d = clone(clf)
            clf_2d.fit(X_train_2d, y_train)

            img_train = create_scatter_plot_2d(X_train_2d, y_train, clf_2d,
                                               f"Decision Boundary Train (PCA) - {model_selected}")
            img_test = create_scatter_plot_2d(X_test_2d, y_test, clf_2d,
                                              f"Decision Boundary Test (PCA) - {model_selected}")

        # Costruzione della risposta con le metriche e i grafici
        response_data = {
            "model_selected": model_selected,
            "validation_metrics": metrics,
            "n_classes": n_classes,
            "graph_train": f"data:image/png;base64,{img_train}",
            "graph_test": f"data:image/png;base64,{img_test}"
        }

        return clean_for_json(response_data)

    except Exception as e:
        logging.error("Errore in predict_test_data: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per l'esecuzione della cross-validation con diversi modelli e strategie

@app.post("/cross_validate")
async def cross_validate_model(request: CrossValidateRequest):
    global global_X, global_y, global_full_numerical_cols, global_full_categorical_cols

    if global_X is None or global_y is None:
        raise HTTPException(status_code=400, detail="Dati non pronti. Caricare e selezionare features/target prima.")

    try:
        # Definizione dei transformer per preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        effective_categorical_cols = [c for c in global_full_categorical_cols if c in global_X.columns]
        effective_numerical_cols = [c for c in global_full_numerical_cols if c in global_X.columns]

        transformers_list = []
        if effective_numerical_cols:
            transformers_list.append(('num', numeric_transformer, effective_numerical_cols))
        if effective_categorical_cols:
            transformers_list.append(('cat', categorical_transformer, effective_categorical_cols))

        if not transformers_list:
            raise HTTPException(status_code=400, detail="Nessuna colonna numerica o categorica trovata per il preprocessing.")

        preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')

        # Inizializzazione del classificatore scelto
        if request.model_selected == "Random Forest":
            clf = RandomForestClassifier(max_depth=5, random_state=42, class_weight='balanced')
        elif request.model_selected == "SVM":
            clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                          random_state=42, class_weight='balanced')
        elif request.model_selected == "XGBoost":
            clf = xgb.XGBClassifier(objective='binary:logistic',
                                    n_estimators=100,
                                    learning_rate=0.1,
                                    max_depth=3,
                                    random_state=42)
        elif request.model_selected == "Neural Network":
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1)
        else:
            raise HTTPException(status_code=400, detail=f"Modello '{request.model_selected}' non supportato.")

        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', clf)])

        # Selezione della strategia di validazione
        if request.validation_type == "k-fold":
            if request.folds <= 1:
                raise HTTPException(status_code=400, detail="K-Fold richiede almeno 2 fold.")
            cv_strategy = KFold(n_splits=request.folds, shuffle=True, random_state=42)
            n_splits = request.folds
        elif request.validation_type == "stratified-k-fold":
            if request.folds <= 1:
                raise HTTPException(status_code=400, detail="Stratified K-Fold richiede almeno 2 fold.")
            min_samples_per_class = np.min(np.bincount(global_y))
            if min_samples_per_class < request.folds:
                raise HTTPException(status_code=400,
                                    detail=f"Impossibile usare Stratified K-Fold con {request.folds} fold. Una classe ha solo {min_samples_per_class} campioni.")
            cv_strategy = StratifiedKFold(n_splits=request.folds, shuffle=True, random_state=42)
            n_splits = request.folds
        elif request.validation_type == "loocv":
            cv_strategy = LeaveOneOut()
            n_splits = len(global_X)
        else:
            raise HTTPException(status_code=400,
                                detail=f"Tipo di validazione '{request.validation_type}' non supportato.")

        # Definizione delle metriche da calcolare
        roc_auc_ovr_scorer = make_scorer(multiclass_roc_auc_score, needs_proba=True)
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'roc_auc_ovr': roc_auc_ovr_scorer
        }

        logging.info(f"Avvio {request.validation_type} con {n_splits} fold per {request.model_selected}...")

        cv_results = cross_validate(
            full_pipeline,
            global_X,
            global_y,
            cv=cv_strategy,
            scoring=scoring_metrics,
            return_train_score=False,
            n_jobs=-1
        )

        logging.info("Cross-validation completata.")

        # Calcolo delle metriche medie e deviazioni standard
        mean_metrics = {
            'accuracy': safe_float(np.nanmean(cv_results['test_accuracy'])),
            'precision': safe_float(np.nanmean(cv_results['test_precision_macro'])),
            'recall': safe_float(np.nanmean(cv_results['test_recall_macro'])),
            'f1_score': safe_float(np.nanmean(cv_results['test_f1_macro'])),
            'roc_auc': safe_float(np.nanmean(cv_results['test_roc_auc_ovr']))
        }

        std_metrics = {
            'accuracy_std': safe_float(np.nanstd(cv_results['test_accuracy'])),
            'precision_std': safe_float(np.nanstd(cv_results['test_precision_macro'])),
            'recall_std': safe_float(np.nanstd(cv_results['test_recall_macro'])),
            'f1_score_std': safe_float(np.nanstd(cv_results['test_f1_macro'])),
            'roc_auc_std': safe_float(np.nanstd(cv_results['test_roc_auc_ovr']))
        }

        # Conversione degli array NumPy in liste per la serializzazione JSON
        cleaned_cv_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cv_results.items()}

        response_data = {
            "model_selected": request.model_selected,
            "validation_type": request.validation_type,
            "folds": n_splits,
            "average_metrics": mean_metrics,
            "std_dev_metrics": std_metrics,
            "all_fold_results": cleaned_cv_results
        }

        logging.debug(f"Restituendo response_data: {response_data}")
        return clean_for_json(response_data)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Errore durante la cross-validation ({request.validation_type} / {request.model_selected}): %s",
                      traceback.format_exc())
        if isinstance(e, ValueError) and "TypeError" in str(e):
            raise HTTPException(status_code=500,
                                detail=f"Errore di serializzazione JSON: Verificare tipi di dati in 'all_fold_results'. Dettagli: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"Errore interno durante la cross-validation: {str(e)}")
# Endpoint per il reset dello stato globale del server

@app.post("/reset_data")
async def reset_data():
    global global_df, X_train, X_test, y_train, y_test
    global categorical_cols, numerical_cols
    global scaler
    global global_X, global_y
    global global_full_numerical_cols, global_full_categorical_cols

    try:
        # Ripristina tutte le variabili globali a uno stato iniziale
        global_df = None
        X_train, X_test, y_train, y_test = None, None, None, None
        categorical_cols, numerical_cols = [], []
        scaler = None
        global_X = None
        global_y = None
        global_full_numerical_cols = []
        global_full_categorical_cols = []

        logging.info("Stato resettato con successo.")
        return {"message": "Stato resettato correttamente."}
    except Exception as e:
        logging.error(f"Errore durante il reset dello stato: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Errore durante il reset: {str(e)}")


# Endpoint per la predizione su una singola istanza

@app.post("/predict_instance")
async def predict_instance(request: InstanceRequest):
    global global_scaler, global_encoder, clf_model
    global selected_independent_features, categorical_cols, numerical_cols
    global global_feature_columns, global_num_imputer, global_cat_imputer

    try:
        if clf_model is None:
            raise HTTPException(status_code=400, detail="Modello non addestrato.")

        input_data = request.instance

        # Verifica che tutte le feature necessarie siano presenti
        missing = [col for col in selected_independent_features if col not in input_data]
        if missing:
            raise HTTPException(status_code=400, detail=f"Feature mancanti: {missing}")

        instance_df = pd.DataFrame([input_data])

        # Conversione dei valori numerici
        for col in numerical_cols:
            instance_df[col] = pd.to_numeric(instance_df[col], errors='coerce')

        # Imputazione e scaling dei dati numerici
        if numerical_cols:
            instance_df[numerical_cols] = global_num_imputer.transform(instance_df[numerical_cols])
            instance_df[numerical_cols] = global_scaler.transform(instance_df[numerical_cols])

        # Imputazione ed encoding dei dati categorici
        if categorical_cols:
            instance_df[categorical_cols] = global_cat_imputer.transform(instance_df[categorical_cols])
            encoded_array = global_encoder.transform(instance_df[categorical_cols])
            encoded_df = pd.DataFrame(encoded_array, columns=global_encoder.get_feature_names_out())
            instance_df = instance_df.drop(columns=categorical_cols).reset_index(drop=True)
            instance_df = pd.concat([instance_df, encoded_df], axis=1)

        # Aggiunta delle feature mancanti con valore 0
        for col in global_feature_columns:
            if col not in instance_df.columns:
                instance_df[col] = 0

        instance_df = instance_df[global_feature_columns]

        # Predizione e probabilità
        prediction = clf_model.predict(instance_df)[0]
        probabilities = clf_model.predict_proba(instance_df)[0]
        n_classes = len(probabilities)

        if n_classes == 2:
            return {
                "prediction": int(prediction),
                "probability_class_0": round(float(probabilities[0]), 4),
                "probability_class_1": round(float(probabilities[1]), 4)
            }
        else:
            return {
                "prediction": int(prediction),
                "probabilities_per_class": {str(i): round(float(prob), 4) for i, prob in enumerate(probabilities)}
            }

    except Exception as e:
        logging.error(f"Errore in /predict_instance: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

