import torch
import numpy as np

class EarlyStopping:
    """Monitoriza a perda de validação e para o treino se não houver melhoria."""
    def __init__(self, patience=2, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict() # Guarda os melhores pesos
        else:
            self.counter += 1
            print(f"   -> EarlyStopping counter: {self.counter} de {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_weights(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)