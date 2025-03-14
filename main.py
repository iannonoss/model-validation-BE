from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from models import PreprocessRequest
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import logging
from pydantic import BaseModel
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, KFold

logging.basicConfig(
    level=logging.DEBUG,  # Imposta livello di log a DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",  # Formato log con timestamp
)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Specifica l'origin corretto
    allow_credentials=True,
    allow_methods=["*"],  # Permette tutti i metodi (GET, POST, ecc.)
    allow_headers=["*"],  # Permette tutti gli header
)

# Creazione della cartella per l'upload dei file
UPLOAD_DIR = "uploads"

# Verifica se la cartella esiste e imposta i permessi
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, mode=0o777, exist_ok=True)
    os.chmod(UPLOAD_DIR, 0o777)  # Rende la cartella disponibile in lettura e in scrittura
global_df = None


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global global_df
    file_location = f"{UPLOAD_DIR}/{file.filename}"

    # Salviamo il file
    with open(file_location, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(file_location)
    global_df = df
    columns = df.columns.tolist()
    missing_values = df.isnull().sum()
    categorical_cols = [cname for cname in df if df[cname].dtype == object]
    numerical_cols = [cname for cname in df if df[cname].dtype in ['int64', 'float64']]
    missing_values = {col: int(val) for col, val in missing_values.items()}
    preview_data = df.head().to_dict(orient="records")

    return {"filename": file.filename,
            "columns": columns,
            "size": os.path.getsize(file_location),
            "missing_values": missing_values,
            "categorical_cols": categorical_cols,
            "numerical_cols": numerical_cols,
            "preview_data": preview_data}

@app.post("/preprocess_data")
async def preprocess_data(request: PreprocessRequest):
    global global_df
    if global_df is None:
        raise HTTPException(status_code=400, detail="Nessun dataset caricato.")

    df = global_df.copy()

    #  Debug: Controlliamo i dati ricevuti
    print("✅ Richiesta ricevuta:", request.dict())

    try:

        #  Controllo se le colonne esistono
        if request.target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Colonna target '{request.target}' non trovata nel dataset.")

        for feature in request.features:
            if feature not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature '{feature}' non trovata nel dataset.")

        X = df[request.features]
        y = df[request.target]

        #  Debug: Verifica i tipi di dati
        print("🔍 Tipi di dati di X:\n", X.dtypes)
        print("🔍 Tipo di dati di y:", y.dtype)

        #  se y ha solo 1 valore (impossibile da splittare)
        if y.nunique() < 2:
            raise HTTPException(status_code=400,
                                detail="Il target ha un solo valore, impossibile fare la classificazione.")

        if X.isnull().all().all():
            raise HTTPException(status_code=400, detail="Tutte le feature selezionate contengono solo valori NaN.")

        #  Splitting dei dati
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )

        # # Identificare colonne numeriche e categoriali
        # categorical_cols = [c for c in X_train.columns if X_train[c].dtype == object]
        # numerical_cols = [c for c in X_train.columns if X_train[c].dtype in ["int64", "float64"]]
        #
        # # Imputazione sui dati di training
        # num_imputer = SimpleImputer(strategy="mean")
        # cat_imputer = SimpleImputer(strategy="most_frequent")
        #
        # X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        # X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        #
        # # Applichiamo la stessa trasformazione sui dati di test
        # X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
        # X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        #
        # # Encoding delle variabili categoriali
        # encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        # X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        # X_test_encoded = encoder.transform(X_test[categorical_cols]) # non rifittare
        #
        # # Creiamo DataFrame per le colonne categoriali
        # X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out())
        # X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out())
        #
        # # Rimuoviamo le colonne originali categoriali e le sostituiamo con le nuove
        # X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
        # X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)
        # X_train = X_train.join(X_train_encoded)
        # X_test = X_test.join(X_test_encoded)
        #
        # # Normalizzazione delle feature numeriche (scaliamo solo i dati di training)
        # scaler = StandardScaler()
        # X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        # X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        return {
            "split_type": request.split_type,
            "test_size": request.test_size,
            "train_preview": X_train.head().to_dict(orient="records"),
            "valid_preview": X_test.head().to_dict(orient="records"),
        }

    except Exception as e:
        print("❌ ERRORE:", str(e))
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

