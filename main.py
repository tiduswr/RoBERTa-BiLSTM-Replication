from src.config import (
    DEVICE, SEED, DATASET_NAME, HIDDEN_DIM, DROPOUT_PROB, 
    LEARNING_RATE, EPOCHS, TABELAS_ARTIGO,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA
)
from src.utils import set_seed
from src.data_loader import DataModule
from src.dataset_strategies import IMDBStrategy, TwitterAirlineStrategy, Sentiment140Strategy
from src.model import RoBERTaBiLSTM
from src.trainer import ModelTrainer

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
        hidden_dim=HIDDEN_DIM, 
        num_labels=num_classes, 
        dropout_prob=DROPOUT_PROB
    )
    
    runtime_engine = ModelTrainer(
        model=model_hybrid, 
        train_loader=train_loader, 
        val_loader=val_loader,
        test_loader=test_loader, 
        lr=LEARNING_RATE, 
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE
    )
    
    runtime_engine.train()
    runtime_engine.evaluate(target_metrics=TABELAS_ARTIGO[DATASET_NAME], dataset_name=DATASET_NAME)

if __name__ == "__main__":
    main()