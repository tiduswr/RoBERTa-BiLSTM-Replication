# Replicação de Fine-Tuning do BERT para Análise de Sentimentos (SST-2)

Este repositório contém a implementação prática e a replicação do processo de fine-tuning do modelo **BERT** (especificamente o `bert-base-uncased`) utilizando a tarefa **SST-2** (Stanford Sentiment Treebank) do benchmark GLUE. 

O objetivo deste projeto é reproduzir as metodologias e os resultados apresentados no artigo seminal da Google AI Language: *[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)*.
## 🚀 Sobre o Projeto

O BERT introduziu representações bidirecionais profundas a partir de texto não rotulado. No entanto, o artigo demonstra que o poder real do modelo está na sua simplicidade de adaptação para tarefas específicas (*downstream tasks*) com modificações arquiteturais mínimas.

Neste projeto, acoplamos uma cabeça de classificação binária ao estado oculto final do token especial `[CLS]` do BERT para prever se uma crítica de cinema extraída do dataset SST-2 possui um sentimento positivo ou negativo.

### 📊 Desempenho e Resultados Esperados
* **Acurácia Alvo (Artigo Original):** **92.9%** no conjunto de validação para o `BERT BASE`.
* **Métricas Computadas:** Perda de Treinamento (*Training Loss*), Perda de Validação (*Validation Loss*) e Acurácia (*Accuracy*).

## 📦 Pré-requisitos e Instalação

Para evitar conflitos comuns de compilação em bibliotecas de imagem (como o erro de sistema `libtiff.so.5` ou incompatibilidades internas do `Pillow` com o `PyTorch`), a instalação deve ser realizada estritamente via **Conda** utilizando os canais oficiais indexados.

### 1. Clonar o Repositório

```bash
git clone https://github.com/tiduswr/bert-replication-sst2.git
cd bert-replication-sst2
```

### 2. Criar o Ambiente Conda

Navegue até a raiz do projeto (onde o arquivo `environment.yml` está localizado) e execute:

```bash
conda env create -f environment.yml
```

### 3. Ativar o Ambiente

```bash
conda activate bert_env
```

## 🖥️ Como Executar o Código

### Localmente (Com GPU Própria, ex: RTX 4060 Ti)

1. Certifique-se de que o seu ambiente Conda está ativado (`conda activate bert_env`).
2. Inicie o servidor do Jupyter Notebook ou abra a pasta no VS Code:
```bash
jupyter notebook
```

3. Abra o ficheiro `replicacao_bert.ipynb`.
4. Selecione o Kernel correspondente ao seu ambiente `bert_env`.
5. Execute as células sequencialmente. O script detectará automaticamente os seus *Tensor Cores* de hardware através do PyTorch e ativará precisão mista automática (`fp16=True`) para acelerar o treino.

## 📝 Hiperparâmetros Utilizados

Os parâmetros configurados na classe `TrainingArguments` seguem rigorosamente as recomendações da Secção 4 do artigo original para tarefas do GLUE:

* **Taxa de Aprendizado (Learning Rate):** `2e-5`
* **Tamanho do Lote (Batch Size):** `32`
* **Épocas de Treino (Epochs):** `3`
* **Decaimento de Peso (Weight Decay):** `0.01`

## 🤝 Contribuições

Sinta-se à vontade para abrir *Issues* para relatar problemas ou sugerir novos experimentos (como testar o modelo em outras tarefas do GLUE como MNLI ou QNLI). Antes de enviar um *Pull Request*, por favor, leia o arquivo **CONTRIBUTING.md**