import nltk
from nltk.stem import WordNetLemmatizer
import re

# Download dos dicionários de lematização do NLTK (executa apenas na primeira vez)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Limpeza textual com Stop-words específicas e Lematização (Algoritmo 1)."""
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