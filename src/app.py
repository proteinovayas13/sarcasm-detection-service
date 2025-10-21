from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import time
import os

app = FastAPI(
    title="Sarcasm Detection API",
    description="API для определения сарказма в английских текстах",
    version="1.0.0"
)

# Загрузка моделей
try:
    # Используем абсолютные пути
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'lr_model.joblib')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.joblib')

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Модели успешно загружены!")
except Exception as e:
    print(f"❌ Ошибка загрузки отдельных моделей: {e}")
    # Пробуем загрузить package
    try:
        package_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'logreg_tfidf_package.joblib')
        pipeline = joblib.load(package_path)
        model = pipeline['model']
        vectorizer = pipeline['vectorizer']
        print("✅ Package модель успешно загружена!")
    except Exception as e2:
        print(f"❌ Ошибка загрузки package: {e2}")
        raise e2


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    class_name: str
    class_id: int
    probability: float
    confidence: str
    processing_time: float


@app.get("/")
async def root():
    return {
        "message": "Sarcasm Detection API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sarcasm(request: TextRequest):
    start_time = time.time()

    try:
        # Векторизация текста
        text_vectorized = vectorizer.transform([request.text])

        # Предсказание
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]

        # Определение класса и вероятности
        class_name = "Sarcasm" if prediction == 1 else "Not Sarcasm"
        probability = float(probabilities[prediction])

        # Уровень уверенности
        if probability > 0.8:
            confidence = "high"
        elif probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        processing_time = time.time() - start_time

        return PredictionResponse(
            text=request.text,
            class_name=class_name,
            class_id=int(prediction),
            probability=probability,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


@app.get("/model-info")
async def model_info():
    return {
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "classes": ["Not Sarcasm", "Sarcasm"],
        "features": vectorizer.get_feature_names_out().shape[0] if hasattr(vectorizer,
                                                                           'get_feature_names_out') else "unknown"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    