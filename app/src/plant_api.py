from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import gc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pasta app
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.tflite")
DATA_PATH = os.path.join(BASE_DIR, "data", "plant_diseases.json")


IMG_SIZE = 224

# Carregar modelo TFLite
print("Carregando modelo TFLite...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Modelo carregado com sucesso!")

# Carregar banco de dados de doenças
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DISEASES_DATA = json.load(f)

# Criar mapa: prediction -> dados
DISEASES_MAP = {item["prediction"]: item for item in DISEASES_DATA}

# Lista de labels (mantém a ordem usada no treinamento do modelo)
LABELS = list(DISEASES_MAP.keys())

# Função para previsão usando TFLite
def predict_disease(image_bytes: bytes) -> str:
    try:
        # Processar imagem
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Rodar predição TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        class_index = int(np.argmax(prediction))

        # Limpeza de memória
        del img_array, img
        gc.collect()

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
        # Devolve o objeto inteiro do JSON
        response = disease_info

    return JSONResponse(content=response)
