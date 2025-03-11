from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

# Inizializziamo FastAPI PRIMA del CORS
app = FastAPI()

# Configuriamo CORS prima di definire gli endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Specifica l'origin corretto
    allow_credentials=True,
    allow_methods=["*"],  # Permette tutti i metodi (GET, POST, ecc.)
    allow_headers=["*"],  # Permette tutti gli header
)

# Test API
@app.get("/")
async def root():
    return {"message": "Backend FastAPI attivo!"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# Creazione della cartella per l'upload dei file
UPLOAD_DIR = "uploads"

# Verifica se la cartella esiste e imposta i permessi
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, mode=0o777, exist_ok=True)  # Assicura che la cartella sia accessibile
    os.chmod(UPLOAD_DIR, 0o777)  # Rende la cartella scrivibile

# Endpoint per il caricamento CSV
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"

    # Salviamo il file
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Leggiamo il CSV per restituire le colonne disponibili
    df = pd.read_csv(file_location)
    columns = df.columns.tolist()

    return {"filename": file.filename, "columns": columns, "size": os.path.getsize(file_location)}
