import sys
import os

# Добавляем src в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_modules():
    print("=== Testing Data Modules ===")
    
    try:
        from data.preprocessing import TextPreprocessor
        from data.dataset_loader import DataLoader
        print("✅ All data imports successful!")
        
        # Тест TextPreprocessor
        preprocessor = TextPreprocessor()
        test_text = "OMG! This is JUST what I needed! 😒 https://example.com"
        processed = preprocessor.preprocess(test_text)
        print(f"✅ TextPreprocessor works: '{test_text}' -> '{processed}'")
        
        # Тест DataLoader
        loader = DataLoader()
        print("✅ DataLoader works!")
        
        print("\n🎉 All data module tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_modules()