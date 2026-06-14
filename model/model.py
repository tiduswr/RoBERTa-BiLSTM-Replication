import torch
import torch.nn as nn
from transformers import RobertaModel
from model.config import MAX_LENGTH, DROPOUT_PROB, HIDDEN_DIM

class RoBERTaBiLSTM(nn.Module):
    """Arquitetura Híbrida Estrita: RoBERTa -> Dropout -> BiLSTM -> Flatten -> Dense 1 -> Dense 2."""
    
    def __init__(self, num_labels: int = 2):
        
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.lstm = nn.LSTM(768, HIDDEN_DIM, batch_first=True, bidirectional=True)
        
        # O artigo exige explicitamente uma camada Flatten
        self.flatten = nn.Flatten()
        
        # O tensor achatado terá o tamanho: MAX_LENGTH * (hidden_dim * 2 direções)
        flattened_size = MAX_LENGTH * (HIDDEN_DIM * 2)
        
        # O artigo exige duas camadas densas (FC)
        self.fc1 = nn.Linear(flattened_size, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, num_labels)

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