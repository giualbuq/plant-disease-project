# 🌿 Plant Disease Detection API

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Descrição

Esta API identifica doenças em plantas utilizando um **modelo TFLite de aprendizado de máquina**.  
Ela recebe imagens de folhas e retorna a doença detectada, com informações detalhadas como nome da planta, descrição e severidade.

---

## 🧠 Tecnologias Utilizadas

- **Python 3.13**  
- **FastAPI**: Framework para criação de APIs rápidas e eficientes.  
- **TensorFlow Lite**: Para inferência leve do modelo de aprendizado profundo.  
- **PIL / Pillow**: Processamento de imagens.  
- **NumPy**: Manipulação de arrays e imagens.  
- **JSON**: Armazenamento de dados de doenças.

---

## 🚀 Como Rodar Localmente

1. **Clone o repositório:**

```bash
git clone https://github.com/giualbuq/plant-disease-project.git
cd plant-disease-project
```

2. **Crie e ative um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4; **Execute a API:**

```bash
uvicorn plant_api:app --reload
```
- A API estará disponível em http://127.0.0.1:8000
- A documentação interativa do Swagger pode ser acessada em http://127.0.0.1:8000/docs

---

## 📦 Endpoints
GET /
Retorna uma mensagem de boas-vindas:

```json
{
  "message": "API de Detecção de doenças de Plantas!"
}
```

POST /predict/
- Recebe uma imagem de planta e retorna a predição com detalhes da doença.
- Parâmetro: file (arquivo de imagem)
- Tipo de arquivo válido: qualquer tipo de imagem (image/*)
Exemplo de retorno JSON:

```json
{
  "prediction": "Tomato___Target_Spot",
  "plant": "Tomato",
  "disease": "Target Spot",
  "description": "Doença comum em tomates que causa manchas circulares nas folhas.",
  "severity": "Moderada"
}
```
Caso a doença não seja encontrada no banco de dados, o retorno será:
```json
{
  "prediction": "Desconhecido",
  "plant": "Desconhecido",
  "disease": "Desconhecido",
  "description": "Não disponível",
  "severity": "Não classificado"
}
```

---

## 📁 Estrutura do Projeto

```text
plant-disease-project
├── .git                                             # Pasta do Git
├── .idea                                            # Configurações do VSCode/IDE
├── .venv                                            # Ambiente virtual Python
├── __pycache__                                      # Cache do Python
├── app                                              # Código principal da aplicação
│   ├── __init__.py                                  # Inicializa o módulo app
│   ├── __pycache__                                  # Cache da pasta app
│   ├── data                                         # Dados da aplicação
│   │   └── plant_diseases.json                      # Banco de dados das doenças
│   ├── model                                        # Modelos treinados
│   │   ├── .gitignore                               # Ignora arquivos do Git
│   │   └── plant_disease_model.tflite               # Modelo TFLite
│   └── src                                          # Código da API
│       ├── __init__.py                              # Inicializa o módulo src
│       ├── __pycache__                              # Cache da pasta src
│       ├── plant_api.py                             # API FastAPI que processa imagens e retorna predição
│       └── requirements.txt                         # Dependências específicas da src
├── requirements.txt                                 # Dependências do projeto
├── run.py                                           # Script para rodar a API localmente
└── runtime.txt                                      # Versão do Python para deploy (ex: Render/Heroku)
```
