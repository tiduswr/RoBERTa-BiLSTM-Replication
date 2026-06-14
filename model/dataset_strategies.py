from typing import Tuple, Protocol
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
from model.config import MAX_LENGTH, HF_TOKEN, DATASET_MAP_CPU_CORE_NUMBER
from model.utils import clean_text
from huggingface_hub import login

if HF_TOKEN and HF_TOKEN != "":
    login(token=HF_TOKEN, add_to_git_credential=False)

class DatasetStrategy(Protocol):
    # Agora a interface exige que se devolvam os 3 Datasets e o num_labels
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset, int]:
        ...

class DatasetStrategy(Protocol):
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset, int]:
        ...

class IMDBStrategy:
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset, int]:
        print("A carregar e a processar o dataset: IMDb...")
        dataset = load_dataset("stanfordnlp/imdb")
        full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
        
        def tokenize_fn(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
            
        split_90_10 = full_dataset.train_test_split(test_size=0.1, seed=42)
        split_5_5 = split_90_10["test"].train_test_split(test_size=0.5, seed=42)
        
        # APLICAÇÃO DO DATASET_MAP_CPU_CORE_NUMBER
        encoded_train = split_90_10["train"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=["text"])
        encoded_val = split_5_5["train"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=["text"])
        encoded_test = split_5_5["test"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=["text"])
        
        return encoded_train, encoded_val, encoded_test, 2

class TwitterAirlineStrategy:
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset, int]:
        print("A carregar e a processar o dataset: Twitter US Airline...")
        dataset = load_dataset("osanseviero/twitter-airline-sentiment")
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        
        def tokenize_fn(batch):
            tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
            tokens["label"] = [label_map[l] for l in batch["airline_sentiment"]]
            return tokens
            
        split_90_10 = dataset["train"].train_test_split(test_size=0.1, seed=42)
        split_5_5 = split_90_10["test"].train_test_split(test_size=0.5, seed=42)
        
        cols = dataset["train"].column_names
        encoded_train = split_90_10["train"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=cols)
        encoded_val = split_5_5["train"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=cols)
        encoded_test = split_5_5["test"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=cols)
        
        return encoded_train, encoded_val, encoded_test, 3

class Sentiment140Strategy:
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset, int]:
        print("A carregar e a processar o dataset: Sentiment140...")
        dataset = load_dataset("stanfordnlp/sentiment140")
        
        def tokenize_fn(batch):
            tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
            tokens["label"] = [1 if l == 4 else 0 for l in batch["sentiment"]]
            return tokens
        
        split_90_10 = dataset["train"].train_test_split(test_size=0.1, seed=42)
        split_5_5 = split_90_10["test"].train_test_split(test_size=0.5, seed=42)
        
        cols = dataset["train"].column_names
        encoded_train = split_90_10["train"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=cols)
        encoded_val = split_5_5["train"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=cols)
        encoded_test = split_5_5["test"].map(tokenize_fn, batched=True, num_proc=DATASET_MAP_CPU_CORE_NUMBER, remove_columns=cols)
        
        return encoded_train, encoded_val, encoded_test, 2
    
class DummyStrategy:
    def prepare_data(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset, int]:
        data = {"text": ["Este é um teste de fumo."] * 200, "label": [1] * 200}
        dataset = Dataset.from_dict(data)
        def tokenize_fn(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        
        split_90_10 = dataset.train_test_split(test_size=0.2, seed=42)
        split_5_5 = split_90_10["test"].train_test_split(test_size=0.5, seed=42)
        
        encoded_train = split_90_10["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
        encoded_val = split_5_5["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
        encoded_test = split_5_5["test"].map(tokenize_fn, batched=True, remove_columns=["text"])
        
        return encoded_train, encoded_val, encoded_test, 2