from typing import Tuple
from model.dataset_strategies import DatasetStrategy
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from model.config import BATCH_SIZE

class DataModule:
    """Gere a formatação PyTorch e os DataLoaders utilizando injeção de dependência."""
    def __init__(self, strategy: DatasetStrategy):
        self.strategy = strategy
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
    def load_and_prepare(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        # Recebe os 3 conjuntos de dados
        train_set, val_set, test_set, num_labels = self.strategy.prepare_data(self.tokenizer)
        
        # Formata para Tensores do PyTorch
        train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        # Cria os iteradores de lote (Batch)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
        
        return train_loader, val_loader, test_loader, num_labels