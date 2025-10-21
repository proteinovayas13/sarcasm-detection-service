import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Callable
import logging

# Скачивание необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Класс для предобработки текстовых данных"""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов и нормализация"""
        if not isinstance(text, str):
            return ""
            
        try:
            # Удаление URL
            text = re.sub(r'http\S+', '', text)
            # Удаление упоминаний и хэштегов
            text = re.sub(r'@\w+|#\w+', '', text)
            # Удаление специальных символов, оставляя буквы и базовую пунктуацию
            text = re.sub(r'[^a-zA-Z\s!?]', '', text)
            # Приведение к нижнему регистру
            text = text.lower()
            # Удаление лишних пробелов
            text = ' '.join(text.split())
            return text
        except Exception as e:
            self.logger.error(f"Error cleaning text: {e}")
            return ""
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Удаление стоп-слов"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Лемматизация токенов"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, steps: List[Callable] = None) -> str:
        """Полный пайплайн предобработки текста"""
        if steps is None:
            steps = [self.clean_text, word_tokenize, 
                    self.remove_stopwords, self.lemmatize_tokens]
        
        try:
            processed = text
            for step in steps:
                if step == word_tokenize:
                    processed = step(processed)
                else:
                    processed = step(processed)
            
            return ' '.join(processed) if isinstance(processed, list) else processed
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {e}")
            return ""