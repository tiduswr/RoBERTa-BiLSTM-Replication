from src.config import DEVICE
from src.dataset_strategies import DummyStrategy
from src.data_loader import DataModule
from src.model import RoBERTaBiLSTM
from src.trainer import ModelTrainer
from torch.utils.data import DataLoader, Subset

def run_smoke_test():
    print(f"🚀 Iniciando SMOKE TEST no dispositivo: {DEVICE}")
    
    data_module = DataModule(strategy=DummyStrategy())
    train_loader, val_loader, test_loader, num_classes = data_module.load_and_prepare()
    
    # Reduzir para apenas 1 batch
    smoke_train = Subset(train_loader.dataset, range(16))
    smoke_val = Subset(val_loader.dataset, range(16))
    smoke_test = Subset(test_loader.dataset, range(16))
    
    # Loaders minusculos
    train_loader = DataLoader(smoke_train, batch_size=2, drop_last=True)
    val_loader = DataLoader(smoke_val, batch_size=2, drop_last=True)
    test_loader = DataLoader(smoke_test, batch_size=2, drop_last=True)
    
    model = RoBERTaBiLSTM(num_labels=num_classes)
    trainer = ModelTrainer(model, train_loader, val_loader, test_loader, is_smoke_test=True)
    
    # Forçar o trainer a rodar apenas 1 época para o teste não demorar
    trainer.epochs = 1 
    
    print("✅ Smoke test: Executando ciclo de treino e avaliação...")
    trainer.train()
    trainer.evaluate()
    print("✅ Smoke test concluído com sucesso!")

if __name__ == "__main__":
    run_smoke_test()