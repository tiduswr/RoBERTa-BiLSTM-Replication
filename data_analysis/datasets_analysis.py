import os
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from dotenv import load_dotenv

def setup_environment():
    """Carrega variáveis de ambiente e autentica na Hugging Face."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

def get_twitter_distribution(seed=42):
    print("-> A processar Twitter US Airline...")
    dataset = load_dataset("osanseviero/twitter-airline-sentiment")["train"]
    train_split = dataset.train_test_split(test_size=0.1, seed=seed)["train"]
    counts = Counter(train_split["airline_sentiment"])
    return [counts["negative"], counts["neutral"], counts["positive"]]

def get_imdb_distribution(seed=42):
    print("-> A processar IMDb...")
    dataset = load_dataset("stanfordnlp/imdb")
    full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    train_split = full_dataset.train_test_split(test_size=0.1, seed=seed)["train"]
    counts = Counter(train_split["label"])
    return [counts[0], counts[1]]

def get_sentiment140_distribution(seed=42):
    print("-> A processar Sentiment140...")
    dataset = load_dataset("stanfordnlp/sentiment140")["train"]
    train_split = dataset.train_test_split(test_size=0.1, seed=seed)["train"]
    counts = Counter(train_split["sentiment"])
    return [counts[0], counts[4]]

def fetch_all_distributions():
    """Coordena o download e a extração das classes de todos os datasets."""
    print("⏳ A carregar datasets da Hugging Face e a calcular distribuições reais...")
    tw_counts = get_twitter_distribution()
    imdb_counts = get_imdb_distribution()
    s140_counts = get_sentiment140_distribution()
    print("✅ Contagens calculadas com sucesso!\n")
    return tw_counts, imdb_counts, s140_counts

def build_chart_config(tw_counts, imdb_counts, s140_counts):
    """Retorna o dicionário de configuração visual para os 3 sub-gráficos."""
    return {
        "Twitter US Airline": {
            "title": "Twitter US Airline\n(Treino - 90%)",
            "classes": ["negativo", "neutro", "positivo"],
            "counts": tw_counts,
            "colors": ["#00c2c7", "#bf00bf", "#c0c000"],
            "subtitle": "(a) Twitter US Airline"
        },
        "IMDB Dataset": {
            "title": "Dataset IMDb\n(Treino - 90%)",
            "classes": ["negativo", "positivo"],
            "counts": imdb_counts,
            "colors": ["#00c2c7", "#bf00bf"],
            "subtitle": "(b) Avaliações IMDb"
        },
        "Sentiment140 Dataset": {
            "title": "Dataset Sentiment140\n(Treino - 90%)",
            "classes": ["negativo", "positivo"],
            "counts": s140_counts,
            "colors": ["#00c2c7", "#bf00bf"],
            "subtitle": "(c) Sentiment140"
        }
    }

def annotate_bars(ax, bars, max_count):
    """Adiciona os valores numéricos no topo de cada barra com formatação PT-BR."""
    for bar in bars:
        yval = bar.get_height()
        yval_str = f"{int(yval):,}".replace(",", ".")
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + (max_count * 0.02), 
            yval_str, 
            ha='center', va='bottom', 
            fontsize=11, fontweight='bold', color='#333333'
        )

def style_axis(ax, data):
    """Aplica formatação padronizada de títulos, labels e grelha ao eixo."""
    ax.set_title(data["title"], fontsize=14, pad=15, fontweight='bold')
    ax.set_ylabel("Número de comentários", fontsize=12)
    ax.set_xlabel(f"Classes\n\n{data['subtitle']}", fontsize=13, labelpad=15)
    
    ax.grid(axis='y', linestyle='-', color='#d3d3d3', alpha=0.8)
    ax.set_axisbelow(True)
    
    # Define as categorias no eixo X de forma limpa
    ax.set_xticks(range(len(data["classes"])))
    ax.set_xticklabels(data["classes"], rotation=25, ha="right", fontsize=12, rotation_mode="anchor")
    
    # Dá um respiro de 15% no topo para os números não cortarem
    ax.set_ylim(0, max(data["counts"]) * 1.15)

def plot_distributions(datasets_info, output_file="reproducibility_dataset_distribution_ptbr.png"):
    """Constrói a figura iterando sobre as configurações e exporta a imagem."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (dataset_name, data) in zip(axes, datasets_info.items()):
        bars = ax.bar(
            data["classes"], 
            data["counts"], 
            color=data["colors"], 
            width=0.6,
            edgecolor="white", 
            linewidth=0.5
        )
        
        annotate_bars(ax, bars, max(data["counts"]))
        style_axis(ax, data)

    plt.tight_layout(pad=3.0)
    plt.show()

# ==========================================
# 5. Ponto de Entrada (Main Execution)
# ==========================================
def main():
    setup_environment()
    tw_counts, imdb_counts, s140_counts = fetch_all_distributions()
    datasets_info = build_chart_config(tw_counts, imdb_counts, s140_counts)
    plot_distributions(datasets_info)

if __name__ == "__main__":
    main()