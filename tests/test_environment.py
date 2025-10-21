import sys
import os

# Добавление src в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_environment():
    print("=== Тестирование окружения Sarcasm Detection ===")
    
    # Тест импортов
    try:
        from data.preprocessing import TextPreprocessor
        from data.dataset_loader import DataLoader
        print("✅ Все модули импортируются успешно")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    
    # Тест предобработки
    try:
        preprocessor = TextPreprocessor()
        test_text = "OMG! This is JUST what I needed! 😒 https://example.com"
        processed = preprocessor.preprocess(test_text)
        print(f"✅ Предобработка текста работает: '{test_text}' -> '{processed}'")
    except Exception as e:
        print(f"❌ Ошибка предобработки: {e}")
        return False
    
    # Тест загрузчика данных
    try:
        loader = DataLoader()
        print("✅ DataLoader инициализирован успешно")
    except Exception as e:
        print(f"❌ Ошибка DataLoader: {e}")
        return False
    
    print("\n🎉 Все тесты пройдены! Окружение настроено правильно.")
    return True

if __name__ == "__main__":
    test_environment()