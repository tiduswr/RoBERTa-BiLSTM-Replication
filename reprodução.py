# ==============================================================================
# MÓDULO 0: DEPENDÊNCIAS E CONFIGURAÇÕES GERAIS
# ==============================================================================
# Instalação caso necessário: !pip install -q transformers datasets scikit-learn tqdm
import os
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Definição de Hiperparâmetros Globais Estritos do Artigo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128       # Comprimento máximo de sequência padrão
BATCH_SIZE = 16       # Tamanho do lote estabelecido nos testes
EPOCHS = 5            # Treino limitado a exatamente 5 épocas
LEARNING_RATE = 1e-5  # Taxa de aprendizagem ideal (l = 0.00001)
HIDDEN_DIM = 256      # Unidades ocultas da BiLSTM (h = 256)
DROPOUT_PROB = 0.1    # Taxa de dropout (d = 0.1)

print(f"Ambiente configurado. Dispositivo de execução: {DEVICE}\n")


# ==============================================================================
# MÓDULO 1: PRÉ-PROCESSAMENTO E PREPARAÇÃO DE DADOS
# ==============================================================================
def clean_text(text):
    """
    Executa a limpeza textual exigida no pipeline do artigo:
    Conversão para minúsculas, remoção de URLs, menções/hashtags, pontuação e números.
    """
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Limpeza de URLs
    text = re.sub(r'@[^\s]+|#', '', text)               # Remoção de tags/mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)             # Mantém apenas caracteres alfabéticos
    text = re.sub(r'\s+', ' ', text).strip()           # Colapsa espaços extras
    return text

class DataModule:
    """
    Gere o carregamento, limpeza, tokenização nativa do RoBERTa e a 
    preparação dos DataLoaders para qualquer uma das 3 bases do artigo.
    """
    def __init__(self, dataset_name="imdb"):
        self.dataset_name = dataset_name
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
    def load_and_prepare(self):
        print(f"A carregar e a processar o dataset: {self.dataset_name}...")
        
        # CENÁRIO 1: IMDb (2 classes)
        if self.dataset_name == "imdb":
            dataset = load_dataset("stanfordnlp/imdb")
            num_labels = 2
            
            def tokenize_fn(batch):
                cleaned = [clean_text(t) for t in batch["text"]]
                return self.tokenizer(cleaned, padding="max_length", truncation=True, max_length=MAX_LENGTH)
                
            encoded = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
            train_set, test_set = encoded["train"], encoded["test"]
            
        # CENÁRIO 2: Twitter US Airline (3 classes)
        elif self.dataset_name == "twitter_airline":
            dataset = load_dataset("osanseviero/twitter-airline-sentiment")
            num_labels = 3
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
            
            def tokenize_fn(batch):
                cleaned = [clean_text(t) for t in batch["text"]]
                tokens = self.tokenizer(cleaned, padding="max_length", truncation=True, max_length=MAX_LENGTH)
                tokens["label"] = [label_map[l] for l in batch["airline_sentiment"]]
                return tokens
                
            # Divisão estratificada simulando o artigo (90% treino, 10% teste)
            split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            encoded = split.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
            train_set, test_set = encoded["train"], encoded["test"]
            
        # CENÁRIO 3: Sentiment140 (2 classes)
        elif self.dataset_name == "sentiment140":
            dataset = load_dataset("stanfordnlp/sentiment140")
            num_labels = 2
            
            def tokenize_fn(batch):
                cleaned = [clean_text(t) for t in batch["text"]]
                tokens = self.tokenizer(cleaned, padding="max_length", truncation=True, max_length=MAX_LENGTH)
                tokens["label"] = [1 if l == 4 else 0 for l in batch["sentiment"]]
                return tokens
            
            # Nota: Sentiment140 contém 1.6M de registos. Para replicação exata total use o set inteiro.
            # Se quiser apenas testar o pipeline mais rápido, adicione .select(range(N))
            split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            encoded = split.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
            train_set, test_set = encoded["train"], encoded["test"]
        else:
            raise ValueError("Dataset inválido. Escolha: 'imdb', 'twitter_airline' ou 'sentiment140'.")
            
        # Formatação para Tensores PyTorch
        train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader, num_labels


# ==============================================================================
# MÓDULO 2: ARQUITETURA DO MODELO HÍBRIDO (RoBERTa-BiLSTM)
# ==============================================================================
class RoBERTaBiLSTM(nn.Module):
    """
    Implementação fiel da Arquitetura Proposta:
    RoBERTa-base -> Dropout -> BiLSTM -> Camada Densa de Classificação Linear.
    """
    def __init__(self, hidden_dim=256, num_labels=2, dropout_prob=0.1):
        super().__init__()
        # Extrator de Contexto Base (Transformers)
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        # Camada de Dropout de regularização intermédia
        self.dropout = nn.Dropout(dropout_prob)
        
        # Camada Recorrente Bidirecional (768 mapeado para hidden_dim em direções duplas)
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        
        # Mapeamento Final para as classes (hidden_dim * 2 devido à bidirecionalidade)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        # Fluxo com cálculo de gradiente ativo para permitir o Fine-Tuning do bloco RoBERTa
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Aplicação de Dropout
        lstm_inputs = self.dropout(roberta_outputs)
        
        # Processamento sequencial profundo pela BiLSTM
        lstm_outputs, (hn, cn) = self.lstm(lstm_inputs)
        
        # Concatenação do último estado oculto das duas direções (Forward e Backward)
        final_features = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        
        # Projeção linear logits
        logits = self.fc(final_features)
        return logits


# ==============================================================================
# MÓDULO 3: MOTOR DE TREINO E AVALIAÇÃO CIENTÍFICA
# ==============================================================================
class ModelTrainer:
    """
    Encapsula as políticas de otimização AdamW, cálculo de perdas
    e extração de métricas de validação idênticas às tabelas do artigo.
    """
    def __init__(self, model, train_loader, test_loader, lr=1e-5, epochs=5):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
    def train(self):
        print("\nA iniciar o ciclo de fine-tuning ponta-a-ponta...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Época {epoch+1}/{self.epochs}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / (progress_bar.n if progress_bar.n else 1)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                
    def evaluate(self, target_metrics):
        print("\nA executar a validação final no conjunto de teste...")
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="A extrair predições"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].cpu().numpy()
                
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
        # Cálculo exato usando pesos ponderados (Weighted) como no artigo científico
        acc = accuracy_score(all_labels, all_preds) * 100
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        
        # Exibição do Quadro de Validação Comparativa
        print("\n" + "="*65)
        print("          QUADRO COMPARATIVO: OBTIDO VS ARTIGO CIENTÍFICO")
        print("="*65)
        print(f"Métrica     | Obtido neste script | Alvo Publicado no Artigo")
        print("-"*65)
        print(f"Acurácia    |       {acc:.2f}%       |         {target_metrics['acc']:.2f}%")
        print(f"Precisão    |       {prec:.2f}%       |         {target_metrics['prec']:.2f}%")
        print(f"Recall      |       {rec:.2f}%       |         {target_metrics['rec']:.2f}%")
        print(f"F1-Score    |       {f1:.2f}%       |         {target_metrics['f1']:.2f}%")
        print("="*65 + "\n")


# ==============================================================================
# EXECUÇÃO PRINCIPAL (MAIN LOOP)
# ==============================================================================
if __name__ == "__main__":
    # >>> CONFIGURAÇÃO DO TESTE: Altere a string abaixo para o dataset que quer testar <<<
    # Opções válidas: "imdb" | "twitter_airline" | "sentiment140"
    DATASET_ALVO = "imdb" 
    
    # Dicionário de Metricas de Sucesso Extraídas Diretamente do Artigo Técnico
    TABELAS_ARTIGO = {
        "imdb": {"acc": 92.36, "prec": 92.46, "rec": 92.36, "f1": 92.35},
        "twitter_airline": {"acc": 80.74, "prec": 81.65, "rec": 80.74, "f1": 80.93},
        "sentiment140": {"acc": 82.25, "prec": 82.26, "rec": 82.25, "f1": 82.25}
    }
    
    # 1. Pipeline de Dados
    data_module = DataModule(dataset_name=DATASET_ALVO)
    train_loader, test_loader, num_classes = data_module.load_and_prepare()
    
    # 2. Pipeline de Arquitetura
    model_hybrid = RoBERTaBiLSTM(hidden_dim=HIDDEN_DIM, num_labels=num_classes, dropout_prob=DROPOUT_PROB)
    
    # 3. Pipeline de Otimização e Resultados
    runtime_engine = ModelTrainer(
        model=model_hybrid, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        lr=LEARNING_RATE, 
        epochs=EPOCHS
    )
    runtime_engine.train()
    runtime_engine.evaluate(target_metrics=TABELAS_ARTIGO[DATASET_ALVO])