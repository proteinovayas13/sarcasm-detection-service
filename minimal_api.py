from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title="Sarcasm Detection API",
    description="API для определения сарказма в английских текстах с помощью BERT",
    version="1.0.0"
)

# Модели данных
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    is_sarcasm: bool
    probability: float
    class_name: str

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]

# Эндпоинты
@app.get("/")
async def root():
    return {
        "message": "Sarcasm Detection Service", 
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sarcasm-detection"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: TextRequest):
    """Определение сарказма в одном тексте"""
    logger.info(f"Получен запрос для текста: {request.text}")
    
    # TODO: Заменить на реальную модель BERT
    # Пока используем заглушку для тестирования
    sarcasm_keywords = ['great', 'love', 'wonderful', 'fantastic', 'perfect']
    has_keyword = any(keyword in request.text.lower() for keyword in sarcasm_keywords)
    
    # Простая логика для демонстрации
    is_sarcasm = has_keyword and len(request.text) > 20
    probability = 0.85 if is_sarcasm else 0.15
    
    result = {
        "text": request.text,
        "is_sarcasm": is_sarcasm,
        "probability": probability,
        "class_name": "Sarcasm" if is_sarcasm else "Not sarcasm"
    }
    
    logger.info(f"Результат предсказания: {result}")
    return result

@app.post("/predict-batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Определение сарказма в нескольких текстах"""
    logger.info(f"Получен batch запрос для {len(request.texts)} текстов")
    
    predictions = []
    for text in request.texts:
        # Используем ту же логику, что и в predict_single
        sarcasm_keywords = ['great', 'love', 'wonderful', 'fantastic', 'perfect']
        has_keyword = any(keyword in text.lower() for keyword in sarcasm_keywords)
        
        is_sarcasm = has_keyword and len(text) > 20
        probability = 0.85 if is_sarcasm else 0.15
        
        predictions.append({
            "text": text,
            "is_sarcasm": is_sarcasm,
            "probability": probability,
            "class_name": "Sarcasm" if is_sarcasm else "Not sarcasm"
        })
    
    return BatchResponse(predictions=predictions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)