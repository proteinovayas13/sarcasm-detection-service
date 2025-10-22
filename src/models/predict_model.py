import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
from pathlib import Path

class SarcasmClassifier:
    """Класс для предсказания сарказма с помощью BERT"""
    
    def __init__(self, model_path: str = "models/best_model"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        self.load_model()
        
    def load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"✅ Модель загружена из {self.model_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    def predict(self, text: str):
        """Предсказание для одного текста"""
        try:
            # Токенизация
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                
            # Результаты
            sarcasm_prob = probabilities[0][1].item()
            is_sarcasm = prediction.item() == 1
            
            return {
                "text": text,
                "is_sarcasm": is_sarcasm,
                "probability": sarcasm_prob,
                "class_name": "Sarcasm" if is_sarcasm else "Not sarcasm"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка предсказания: {e}")
            return {
                "text": text,
                "is_sarcasm": False,
                "probability": 0.0,
                "class_name": "Error",
                "error": str(e)
            }
    
    def predict_batch(self, texts: list):
        """Предсказание для списка текстов"""
        return [self.predict(text) for text in texts]

# Создаем глобальный экземпляр классификатора
classifier = SarcasmClassifier()