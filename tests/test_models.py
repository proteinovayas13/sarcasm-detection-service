import sys
import os

# Добавляем src в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_models():
    print("=== Testing Models Module ===")
    
    try:
        from models.model_factory import ModelFactory
        from models.model_evaluator import ModelEvaluator
        print("✅ All imports successful!")
        
        # Тест ModelFactory
        factory = ModelFactory()
        model = factory.create_model('logistic_regression')
        print("✅ ModelFactory works!")
        
        # Тест ModelEvaluator
        evaluator = ModelEvaluator()
        print("✅ ModelEvaluator works!")
        
        # Проверяем доступные модели
        available_models = factory.get_available_models()
        print(f"✅ Available models: {available_models}")
        
        print("\n🎉 All model tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_models()