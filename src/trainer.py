import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from src.config import DEVICE
from src.early_stopping import EarlyStopping
from src.config import (
    DEVICE, SEED, DATASET_NAME, HIDDEN_DIM, DROPOUT_PROB, 
    LEARNING_RATE, EARLY_STOPPING_MAX_EPOCHS, TABELAS_ARTIGO,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA, BATCH_SIZE, MAX_LENGTH,
    USE_AMP # <-- Importação da nova flag
)

class ModelTrainer:
    """Motor de otimização com ciclo de Validação, Early Stopping e Avaliação final."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, test_loader, is_smoke_test: bool = False):
        self.model = model.to(DEVICE)
        self.is_smoke_test = is_smoke_test
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = EARLY_STOPPING_MAX_EPOCHS
        self.early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        
        # Variáveis para rastreamento no JSON
        self.start_time_str = None
        self.best_epoch = EARLY_STOPPING_MAX_EPOCHS
        
    def train(self) -> None:
        # Regista o tempo exato em que o treino começa
        self.start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nA iniciar o ciclo de fine-tuning (Patience={self.early_stopping.patience}, AMP={USE_AMP})...")
        
        # O GradScaler desativa-se automaticamente se enabled=False
        scaler = GradScaler('cuda', enabled=USE_AMP)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # 1. FASE DE TREINO
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Época {epoch+1}/{self.epochs} [Treino]")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                self.optimizer.zero_grad()
                
                # Autocast respeita a flag do .env dinamicamente
                with autocast('cuda', enabled=USE_AMP):
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                total_train_loss += loss.item()
                avg_loss = total_train_loss / (progress_bar.n if progress_bar.n else 1)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
            # 2. FASE DE VALIDAÇÃO
            self.model.eval()
            total_val_loss = 0
            all_val_preds, all_val_labels = [], []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    labels = batch["label"].to(DEVICE)
                    
                    with autocast('cuda', enabled=USE_AMP):
                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_val_preds.extend(preds)
                    all_val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(self.val_loader)
            val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
            print(f"   -> Validação: Loss: {avg_val_loss:.4f} | Acurácia: {val_acc:.2f}%")
            
            # Atualiza a melhor época se a loss de validação melhorar
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_epoch = epoch + 1
            
            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                print("⚠️ Early Stopping ativado: Parando o treino e restaurando os melhores pesos.")
                self.early_stopping.load_best_weights(self.model)
                break
                
    def evaluate(self) -> None:
        print("\nA executar a avaliação final no conjunto de TESTE...")
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="A extrair predições"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].cpu().numpy()
                
                with autocast('cuda', enabled=USE_AMP):
                    outputs = self.model(input_ids, attention_mask)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
        acc = accuracy_score(all_labels, all_preds) * 100
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        
        if not self.is_smoke_test:
            self._print_report(acc, prec, rec, f1)
            self._save_results_to_json(acc, prec, rec, f1)
        else:
            self._print_smoke_test_report(acc)

    def _print_report(self, acc: float, prec: float, rec: float, f1: float) -> None:
        target = TABELAS_ARTIGO[DATASET_NAME]
        print("\n" + "="*65)
        print("          QUADRO COMPARATIVO FINAL: OBTIDO VS ARTIGO CIENTÍFICO")
        print("="*65)
        print(f"Métrica     | Obtido neste script | Alvo Publicado no Artigo")
        print("-"*65)
        print(f"Acurácia    |       {acc:.2f}%       |         {target['acc']:.2f}%")
        print(f"Precisão    |       {prec:.2f}%       |         {target['prec']:.2f}%")
        print(f"Recall      |       {rec:.2f}%       |         {target['rec']:.2f}%")
        print(f"F1-Score    |       {f1:.2f}%       |         {target['f1']:.2f}%")
        print("="*65 + "\n")
    
    def _print_smoke_test_report(self, acc: float) -> None:
        print("          RELATÓRIO DE SMOKE TEST (VERIFICAÇÃO DE FLUXO)")
        print("="*65)
        print(f"Status      | Pipeline Funcional")
        print(f"Acurácia    | {acc:.2f}% (Dataset Sintético)")
        print("-"*65)
        print("✅ Todos os módulos (Data->Model->Trainer) integrados com sucesso.")

    def _save_results_to_json(self, acc: float, prec: float, rec: float, f1: float) -> None:
        """Salva resultados e todas as configurações carregadas do .env num ficheiro JSON."""
        target = TABELAS_ARTIGO[DATASET_NAME]
        folder_path = "data"
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "experiment_results.json")
        
        # Se start_time_str não foi definido (ex: avaliar sem treinar), define para o momento atual
        start_time = self.start_time_str if self.start_time_str else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Atualização dinâmica do campo optimization
        optimization_status = "AMP" if USE_AMP else "FP32 (No AMP)"
        
        new_record = {
            "timestamp": {
                "start": start_time,
                "finish": finish_time
            },
            "dataset": DATASET_NAME,
            "optimization": optimization_status,
            "hyperparameters": {
                "seed": SEED,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT_PROB,
                "early_stopping": {
                    "max-epochs": EARLY_STOPPING_MAX_EPOCHS,
                    "patience": EARLY_STOPPING_PATIENCE,
                    "delta": EARLY_STOPPING_DELTA
                }
            },
            "best-epoch": self.best_epoch,
            "metrics_obtained": {
                "accuracy": round(acc, 2),
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1_score": round(f1, 2)
            },
            "metrics_target_article": {
                "accuracy": target['acc'],
                "precision": target['prec'],
                "recall": target['rec'],
                "f1_score": target['f1']
            }
        }
        
        # Lógica de persistência
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try: 
                    data = json.load(f)
                except: 
                    data = []
        else: 
            data = []
            
        data.append(new_record)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Experiência registada com todos os parâmetros do .env em: '{file_path}'")