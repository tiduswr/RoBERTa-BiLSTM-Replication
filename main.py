from model.config import (
    DEVICE, SEED, DATASET_NAME
)
from model.utils import set_seed
from model.data_loader import DataModule
from model.dataset_strategies import IMDBStrategy, TwitterAirlineStrategy, Sentiment140Strategy
from model.model import RoBERTaBiLSTM
from model.trainer import ModelTrainer

def get_strategy(dataset_name: str):
    """Factory simples para mapear a string do .env para a Estratégia concreta."""
    strategies = {
        "imdb": IMDBStrategy,
        "twitter_airline": TwitterAirlineStrategy,
        "sentiment140": Sentiment140Strategy
    }
    
    if dataset_name not in strategies:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. Verifique o seu .env.")
    
    return strategies[dataset_name]()

def main():
    print(f"Ambiente configurado. Dispositivo: {DEVICE} | Dataset Alvo: {DATASET_NAME.upper()}\n")
    set_seed(SEED)
    
    strategy = get_strategy(DATASET_NAME)
    
    data_module = DataModule(strategy=strategy)
    train_loader, val_loader, test_loader, num_classes = data_module.load_and_prepare()
    
    model_hybrid = RoBERTaBiLSTM( 
        num_labels=num_classes
    )
    
    runtime_engine = ModelTrainer(
        model=model_hybrid, 
        train_loader=train_loader, 
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    runtime_engine.train()
    runtime_engine.evaluate()

if __name__ == "__main__":
    main()