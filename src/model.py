import torch
import torch.nn as nn
from transformers import RobertaModel
from src.config import MAX_LENGTH

class RoBERTaBiLSTM(nn.Module):
    """Arquitetura Híbrida Estrita: RoBERTa -> Dropout -> BiLSTM -> Flatten -> Dense 1 -> Dense 2."""
    
    def __init__(self, hidden_dim: int = 256, num_labels: int = 2, dropout_prob: float = 0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        
        # O artigo exige explicitamente uma camada Flatten
        self.flatten = nn.Flatten()
        
        # O tensor achatado terá o tamanho: MAX_LENGTH * (hidden_dim * 2 direções)
        flattened_size = MAX_LENGTH * (hidden_dim * 2)
        
        # O artigo exige duas camadas densas (FC)
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_inputs = self.dropout(roberta_outputs)
        
        # Captura de toda a sequência
        lstm_outputs, _ = self.lstm(lstm_inputs)
        
        # Achatar o tensor (Flatten)
        flat_out = self.flatten(lstm_outputs)
        
        # Passagem pelas duas camadas Densas
        dense1_out = torch.relu(self.fc1(flat_out))
        logits = self.fc2(dense1_out)
        
        # Nota: A função Softmax é aplicada internamente pela CrossEntropyLoss do PyTorch
        return logits