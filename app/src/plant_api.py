from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import json

# Caminhos
MODEL_PATH = os.path.join("app", "model", "plant_disease_prediction_model.h5")
DATA_PATH = os.path.join("app", "data", "plant_diseases.json")

IMG_SIZE = 224

# Carregar modelo
print("Carregando modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso!")

# Carregar banco de dados de doenças
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DISEASES_DATA = json.load(f)

# Criar mapa: prediction -> dados
DISEASES_MAP = {item["prediction"]: item for item in DISEASES_DATA}

# Lista de labels (mantém a ordem usada no treinamento do modelo)
LABELS = list(DISEASES_MAP.keys())

# Função para previsão
def predict_disease(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))
        return LABELS[class_index]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {str(e)}")

# Criar API
app = FastAPI(title="API de Detecção de Doenças em Plantas")

@app.get("/")
def root():
    return {"message": "API de Detecção de doenças de Plantas!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validar tipo de arquivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem.")

    # Ler bytes da imagem
    image_bytes = await file.read()

    # Fazer predição
    result = predict_disease(image_bytes)

    # Buscar informações completas da doença
    disease_info = DISEASES_MAP.get(result)

    if not disease_info:
        # Caso não encontre, retorna apenas a predição
        response = {
            "prediction": result,
            "plant": "Desconhecido",
            "disease": "Desconhecido",
            "description": "Não disponível",
            "severity": "Não classificado"
        }
    else:
        # 🔥 devolve o objeto inteiro do JSON
        response = disease_info

    return JSONResponse(content=response)

