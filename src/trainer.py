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

class ModelTrainer:
    """Motor de otimização com ciclo de Validação, Early Stopping e Avaliação final."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, test_loader, lr: float, epochs: int, patience: int = 2, delta: int = 0):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.early_stopping = EarlyStopping(patience=patience, delta=delta)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
    def train(self) -> None:
        print(f"\nA iniciar o ciclo de fine-tuning (Patience={self.early_stopping.patience})...")
        scaler = GradScaler('cuda')
        
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
                
                with autocast('cuda'):
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
                    
                    with autocast('cuda'):
                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_val_preds.extend(preds)
                    all_val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(self.val_loader)
            val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
            print(f"   -> Validação: Loss: {avg_val_loss:.4f} | Acurácia: {val_acc:.2f}%")
            
            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                print("⚠️ Early Stopping ativado: Parando o treino e restaurando os melhores pesos.")
                self.early_stopping.load_best_weights(self.model)
                break
                
    def evaluate(self, target_metrics: dict, dataset_name: str) -> None:
        print("\nA executar a avaliação final no conjunto de TESTE...")
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="A extrair predições"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].cpu().numpy()
                
                with autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
        acc = accuracy_score(all_labels, all_preds) * 100
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        
        self._print_report(acc, prec, rec, f1, target_metrics)
        self._save_results_to_json(acc, prec, rec, f1, target_metrics, dataset_name)

    def _print_report(self, acc: float, prec: float, rec: float, f1: float, target: dict) -> None:
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

    def _save_results_to_json(self, acc: float, prec: float, rec: float, f1: float, target: dict, dataset_name: str) -> None:
        folder_path = "data"
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "experiment_results.json")
        
        new_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_name,
            "metrics_obtained": {"accuracy": round(acc, 2), "f1_score": round(f1, 2)},
            "metrics_target_article": {"accuracy": target['acc'], "f1_score": target['f1']}
        }
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try: data = json.load(f)
                except: data = []
        else: data = []
            
        data.append(new_record)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)