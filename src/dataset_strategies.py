from typing import Tuple, Protocol
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
from src.config import MAX_LENGTH
from src.utils import clean_text

class DatasetStrategy(Protocol):
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, int]:
        ...

class IMDBStrategy:
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, int]:
        print("A carregar e a processar o dataset: IMDb...")
        dataset = load_dataset("stanfordnlp/imdb")
        
        # 1. Unir as divisões nativas de 25k + 25k para formar o corpus total de 50k
        full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
        
        def tokenize_fn(batch):
            cleaned = [clean_text(t) for t in batch["text"]]
            return tokenizer(cleaned, padding="max_length", truncation=True, max_length=MAX_LENGTH)
            
        # 2. Divisão fiel à Tabela I do Artigo: 90% Treino e 10% Restante
        split_90_10 = full_dataset.train_test_split(test_size=0.1, seed=42)
        
        # 3. Divide os 10% restantes ao meio (5% Validação, 5% Teste)
        split_5_5 = split_90_10["test"].train_test_split(test_size=0.5, seed=42)
        
        train_data = split_90_10["train"]
        test_data = split_5_5["test"]
        
        encoded_train = train_data.map(tokenize_fn, batched=True, remove_columns=["text"])
        encoded_test = test_data.map(tokenize_fn, batched=True, remove_columns=["text"])
        
        return encoded_train, encoded_test, 2

class TwitterAirlineStrategy:
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, int]:
        print("A carregar e a processar o dataset: Twitter US Airline...")
        dataset = load_dataset("osanseviero/twitter-airline-sentiment")
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        
        def tokenize_fn(batch):
            cleaned = [clean_text(t) for t in batch["text"]]
            tokens = tokenizer(cleaned, padding="max_length", truncation=True, max_length=MAX_LENGTH)
            tokens["label"] = [label_map[l] for l in batch["airline_sentiment"]]
            return tokens
            
        # Divisão fiel ao Artigo: 90% Treino e 10% Restante
        split_90_10 = dataset["train"].train_test_split(test_size=0.1, seed=42)
        
        # Divide os 10% restantes ao meio (5% Validação, 5% Teste)
        split_5_5 = split_90_10["test"].train_test_split(test_size=0.5, seed=42)
        
        # Mapeia apenas os subconjuntos alvo para poupar recursos
        train_data = split_90_10["train"]
        test_data = split_5_5["test"]
        
        encoded_train = train_data.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
        encoded_test = test_data.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
        
        return encoded_train, encoded_test, 3

class Sentiment140Strategy:
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, int]:
        print("A carregar e a processar o dataset: Sentiment140...")
        dataset = load_dataset("stanfordnlp/sentiment140")
        
        def tokenize_fn(batch):
            cleaned = [clean_text(t) for t in batch["text"]]
            tokens = tokenizer(cleaned, padding="max_length", truncation=True, max_length=MAX_LENGTH)
            tokens["label"] = [1 if l == 4 else 0 for l in batch["sentiment"]]
            return tokens
        
        # Divisão fiel ao Artigo: 90% Treino e 10% Restante
        split_90_10 = dataset["train"].train_test_split(test_size=0.1, seed=42)
        
        # Divide os 10% restantes ao meio (5% Validação, 5% Teste)
        split_5_5 = split_90_10["test"].train_test_split(test_size=0.5, seed=42)
        
        # Mapeia apenas os subconjuntos alvo para poupar recursos
        train_data = split_90_10["train"]
        test_data = split_5_5["test"]
        
        encoded_train = train_data.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
        encoded_test = test_data.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
        
        return encoded_train, encoded_test, 2