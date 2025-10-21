import sys
import os

def run_tests():
    """Run project import and functionality tests"""
    
    print("=" * 60)
    print("TESTING SARCASM DETECTION PROJECT")
    print("=" * 60)
    
    # Add src to Python path
    project_root = os.path.dirname(__file__)
    src_path = os.path.join(project_root, 'src')
    sys.path.insert(0, src_path)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Main module
    print("\n1. Testing main module...")
    try:
        import src
        version = getattr(src, '__version__', 'not specified')
        print(f"   SUCCESS: src module imported (version: {version})")
        tests_passed += 1
    except Exception as e:
        print(f"   FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Data modules
    print("\n2. Testing data modules...")
    try:
        from src.data import TextPreprocessor, DataLoader
        print("   SUCCESS: Data modules imported")
        tests_passed += 1
    except Exception as e:
        print(f"   FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Create instances
    print("\n3. Testing class instances...")
    try:
        preprocessor = TextPreprocessor()
        data_loader = DataLoader()
        print("   SUCCESS: Class instances created")
        tests_passed += 1
    except Exception as e:
        print(f"   FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Text preprocessing
    print("\n4. Testing text preprocessing...")
    try:
        test_text = "Hello, this is a test! https://example.com #test"
        processed = preprocessor.preprocess(test_text)
        print(f"   SUCCESS: '{test_text}' -> '{processed}'")
        tests_passed += 1
    except Exception as e:
        print(f"   FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Models modules
    print("\n5. Testing models modules...")
    try:
        from src.models import ModelFactory, ModelEvaluator
        print("   SUCCESS: Models modules imported")
        tests_passed += 1
    except Exception as e:
        print(f"   FAILED: {e}")
        tests_failed += 1
    
    # Results
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    if tests_failed == 0:
        print("ALL TESTS PASSED! Project is ready.")
        return True
    else:
        print(f"There are {tests_failed} issues to fix.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)