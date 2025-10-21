from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import joblib
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFactory:
    """Фабрика для создания и управления ML моделями"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
    
    def create_model(self, model_type, **kwargs):
        """Создание модели по типу"""
        model_registry = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'naive_bayes': MultinomialNB
        }
        
        if model_type not in model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_registry[model_type]
        model = model_class(**kwargs)
        self.models[model_type] = model
        
        logger.info(f"Created {model_type} model")
        return model
    
    def train_model(self, model_type, X_train, y_train):
        """Обучение модели"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Create it first.")
        
        logger.info(f"Training {model_type}...")
        self.models[model_type].fit(X_train, y_train)
        logger.info(f"{model_type} trained successfully")
    
    def predict(self, model_type, X):
        """Предсказание с помощью модели"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
        
        return self.models[model_type].predict(X)
    
    def evaluate_model(self, model_type, X_test, y_test):
        """Оценка модели"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
        
        y_pred = self.predict(model_type, X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.metrics[model_type] = metrics
        logger.info(f"{model_type} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save_model(self, model_type, filepath):
        """Сохранение модели"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
        
        joblib.dump(self.models[model_type], filepath)
        logger.info(f"Model {model_type} saved to {filepath}")
    
    def load_model(self, model_type, filepath):
        """Загрузка модели"""
        self.models[model_type] = joblib.load(filepath)
        logger.info(f"Model {model_type} loaded from {filepath}")
    
    def get_available_models(self):
        """Получение списка доступных моделей"""
        return list(self.models.keys())
    
    def get_model_summary(self):
        """Получение сводки по всем моделям"""
        summary = []
        for model_type, metrics in self.metrics.items():
            summary.append({
                'model': model_type,
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0)
            })
        return pd.DataFrame(summary)