from typing import Tuple
from src.dataset_strategies import DatasetStrategy
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from src.config import MAX_LENGTH, BATCH_SIZE

class DataModule:
    """Gere a formatação PyTorch e os DataLoaders utilizando injeção de dependência."""
    def __init__(self, strategy: DatasetStrategy):
        self.strategy = strategy
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
    def load_and_prepare(self) -> Tuple[DataLoader, DataLoader, int]:
        train_set, test_set, num_labels = self.strategy.prepare_data(self.tokenizer)
        
        train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader, num_labels