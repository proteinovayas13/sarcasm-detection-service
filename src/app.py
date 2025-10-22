from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os

app = FastAPI(
    title="Sarcasm Detection API",
    description="API для определения сарказма в текстах с использованием BERT-Tiny",
    version="2.0.0"
)


class BERTPredictor:
    def __init__(self):
        self.model_name = "cointegrated/rubert-tiny2"
        print("⚡ Загрузка BERT-Tiny модели...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            self.model.eval()  # Режим inference
            print("✅ BERT-Tiny модель успешно загружена!")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise e

    def predict(self, text: str):
        # Токенизация текста
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Получаем вероятности
        prob_not_sarcastic = predictions[0][0].item()
        prob_sarcastic = predictions[0][1].item()

        # Определяем класс
        is_sarcastic = prob_sarcastic > prob_not_sarcastic
        probability = prob_sarcastic if is_sarcastic else prob_not_sarcastic
        class_id = 1 if is_sarcastic else 0

        return {
            "is_sarcastic": is_sarcastic,
            "probability": probability,
            "class_id": class_id,
            "probabilities": {
                "not_sarcastic": prob_not_sarcastic,
                "sarcastic": prob_sarcastic
            }
        }


# Инициализация модели
try:
    predictor = BERTPredictor()
    print("🎯 Модель готова к работе!")
except Exception as e:
    print(f"❌ Критическая ошибка инициализации модели: {e}")
    predictor = None


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    class_name: str
    class_id: int
    probability: float
    confidence: str
    processing_time: float
    model: str


@app.get("/")
async def root():
    return {
        "message": "Sarcasm Detection API with BERT-Tiny",
        "version": "2.0.0",
        "model": "rubert-tiny2",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if predictor else "unhealthy",
        "timestamp": time.time(),
        "model_loaded": predictor is not None,
        "model_name": "cointegrated/rubert-tiny2"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sarcasm(request: TextRequest):
    start_time = time.time()

    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        # Получаем предсказание от BERT
        result = predictor.predict(request.text)

        # Определяем уровень уверенности
        probability = result["probability"]
        if probability > 0.8:
            confidence = "high"
        elif probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        processing_time = time.time() - start_time

        return PredictionResponse(
            text=request.text,
            class_name="Sarcasm" if result["is_sarcastic"] else "Not Sarcasm",
            class_id=result["class_id"],
            probability=probability,
            confidence=confidence,
            processing_time=processing_time,
            model="BERT-Tiny"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


@app.get("/model-info")
async def model_info():
    return {
        "model_type": "BERT-Tiny (Transformers)",
        "model_name": "cointegrated/rubert-tiny2",
        "classes": ["Not Sarcasm", "Sarcasm"],
        "max_length": 512,
        "framework": "PyTorch + Transformers"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    