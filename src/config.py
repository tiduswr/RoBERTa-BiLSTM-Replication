import os
import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.getenv("SEED", "42"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EARLY_STOPPING_MAX_EPOCHS = int(os.getenv("EARLY_STOPPING_MAX_EPOCHS", "5"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
DROPOUT_PROB = float(os.getenv("DROPOUT_PROB", "0.1"))
DATASET_NAME = os.getenv("DATASET_NAME", "imdb").lower()
HF_TOKEN = os.getenv("HF_TOKEN", "")
DATASET_MAP_CPU_CORE_NUMBER = int(os.getenv("DATASET_MAP_CPU_CORE_NUMBER", "8"))

# Early Stopping: monitorização de validação
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "2"))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", "0.0"))

# Tabelas de Métricas do Artigo Científico (Alvos)
# Mantêm-se hardcoded porque representam a "verdade terrestre" do artigo publicado
TABELAS_ARTIGO = {
    "imdb": {"acc": 92.36, "prec": 92.46, "rec": 92.36, "f1": 92.35},
    "twitter_airline": {"acc": 80.74, "prec": 80.94, "rec": 80.74, "f1": 80.73},
    "sentiment140": {"acc": 82.25, "prec": 82.25, "rec": 82.25, "f1": 82.25} 
}