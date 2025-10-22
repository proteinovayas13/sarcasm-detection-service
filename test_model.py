import sys
import os
sys.path.append('src')

from models.predict_model import SarcasmClassifier

def test_model():
    """Тестирование модели"""
    classifier = SarcasmClassifier()
    
    test_texts = [
        "Oh great, another meeting that could have been an email",
        "The weather is nice today",
        "I just love waiting in long lines",
        "This is a normal sentence without sarcasm"
    ]
    
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ:")
    print("=" * 50)
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"📝 Текст: {text}")
        print(f"🎯 Результат: {result['class_name']}")
        print(f"📊 Вероятность: {result['probability']:.3f}")
        print("-" * 30)

if __name__ == "__main__":
    test_model()