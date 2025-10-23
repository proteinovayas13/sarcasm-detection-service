import evaluate
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Комплексная оценка моделей классификации"""
    
    def __init__(self):
        self.f1_metric = evaluate.load("f1")
    
    def evaluate_sklearn_model(self, model, texts: List[str], true_labels: List[int]) -> Dict[str, Any]:
        """Оценка sklearn моделей"""
        start_time = time.time()
        
        if hasattr(model, 'predict_proba'):
            pred_probs = model.predict_proba(texts)
            pred_labels = np.argmax(pred_probs, axis=1)
        else:
            pred_labels = model.predict(texts)
            pred_probs = None
        
        inference_time = time.time() - start_time
        
        return self._calculate_metrics(true_labels, pred_labels, pred_probs, inference_time, len(texts))
    
    def _calculate_metrics(self, true_labels, pred_labels, pred_probs, inference_time, n_samples):
        """Расчет метрик"""
        accuracy = self.accuracy_metric.compute(predictions=pred_labels, references=true_labels)
        f1 = self.f1_metric.compute(predictions=pred_labels, references=true_labels, average='weighted')
        precision = self.precision_metric.compute(predictions=pred_labels, references=true_labels, average='weighted')
        recall = self.recall_metric.compute(predictions=pred_labels, references=true_labels, average='weighted')
        
        metrics = {
            'f1_score': f1['f1'],
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / n_samples,
            'predictions': pred_labels,
            'probabilities': pred_probs
        }
        
        return metrics
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """Сравнение всех моделей"""
        comparison_data = []
        
        for model_name, results in models_results.items():
            comparison_data.append({
                'Model': model_name,
                'F1-Score': results['f1_score'],
                'Inference Time (s)': results['inference_time'],
                'Time per Sample (ms)': results['inference_time_per_sample'] * 1000
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, comparison_df: pd.DataFrame):
        """Визуализация сравнения моделей"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Метрики качества
        metrics = [F1-Score']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            sns.barplot(data=comparison_df, x='Model', y=metric, ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Время инференса
        plt.figure(figsize=(10, 6))
        sns.barplot(data=comparison_df, x='Model', y='Time per Sample (ms)')
        plt.title('Inference Time per Sample')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
