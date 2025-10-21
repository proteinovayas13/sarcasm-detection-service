from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import TextPreprocessor

app = FastAPI(title="Sarcasm Detection API", version="1.0.0")

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    processed_text: str
    message: str

# Initialize preprocessor
preprocessor = TextPreprocessor()

@app.get("/")
async def root():
    return {"message": "Sarcasm Detection API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/preprocess", response_model=PredictionResponse)
async def preprocess_text(request: TextRequest):
    """
    Preprocess text - remove URLs, mentions, etc.
    """
    try:
        processed = preprocessor.preprocess(request.text)
        return PredictionResponse(
            processed_text=processed,
            message="Text processed successfully"
        )
    except Exception as e:
        return PredictionResponse(
            processed_text="",
            message=f"Error: {str(e)}"
        )

@app.post("/analyze")
async def analyze_sarcasm(request: TextRequest):
    """
    Simple sarcasm analysis (placeholder for now)
    """
    processed = preprocessor.preprocess(request.text)
    
    # Simple rule-based detection (for demo)
    sarcasm_indicators = ['love waiting', 'great', 'wonderful', 'fantastic', 'just what i needed']
    is_sarcastic = any(indicator in processed.lower() for indicator in sarcasm_indicators)
    
    return {
        "original_text": request.text,
        "processed_text": processed,
        "is_sarcastic": is_sarcastic,
        "confidence": 0.75 if is_sarcastic else 0.25,
        "message": "This is a demo version - ML model will be added later"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
