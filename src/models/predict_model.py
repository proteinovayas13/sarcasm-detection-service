import torch 
from transformers import AutoTokenizer, AutoModel 
 
classifier = None 
tokenizer = None 
model = None 
 
def load_model(): 
    global classifier, tokenizer, model 
    if classifier is None: 
        tokenizer = AutoTokenizer.from_pretrained('./models') 
        model = AutoModel.from_pretrained('./models') 
    return model, tokenizer 
 
def predict_sarcasm(text, model, tokenizer): 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128) 
    with torch.no_grad(): 
        outputs = model(**inputs) 
    embeddings = outputs.last_hidden_state.mean(dim=1) 
    confidence = embeddings.mean().abs().item() 
    is_sarcasm = confidence 
    return {'text': text, 'is_sarcasm': bool(is_sarcasm), 'confidence': confidence} 
