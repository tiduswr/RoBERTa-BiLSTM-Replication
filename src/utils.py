import os
import re
import torch
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer

# Download dos dicionários de lematização do NLTK (executa apenas na primeira vez)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

def set_seed(seed: int = 42) -> None:
    """Fixa as sementes aleatórias para garantir reprodutibilidade (Open Science)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def clean_text(text: str) -> str:
    """Limpeza textual com Stop-words específicas e Lematização (Algoritmo 1 do artigo)."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[^\s]+|#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Eliminação de Stop-Words específicas listadas no artigo
    stop_words = {"the", "an", "a"}
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    # Aplicação de Lematização
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return ' '.join(words).strip()