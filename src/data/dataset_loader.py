import json
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

class DataLoader:
    """Класс для загрузки и разделения данных"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_json_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка данных из JSON файла"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            df = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame, target_column: str, 
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Tuple:
        """Разделение данных на train/validation/test"""
        try:
            # Сначала разделяем на train+val и test
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Затем разделяем train+val на train и val
            val_relative_size = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative_size, 
                random_state=random_state, stratify=y_temp
            )
            
            self.logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise