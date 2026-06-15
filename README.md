# Replicação Científica: RoBERTa-BiLSTM para Análise de Sentimentos

[![Open Science](https://img.shields.io/badge/Open%20Science-Reproducible-brightgreen)](#)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](#)
[![PyTorch 12.4](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C)](#)

Este repositório contém a implementação independente, modular e reprodutível do modelo híbrido proposto no artigo original publicado na *IEEE Transactions on Emerging Topics in Computational Intelligence*. A arquitetura une a extração de contexto de Large Language Models (LLMs) com o processamento sequencial profundo de Redes Neurais Recorrentes (RNNs).

> **Artigo Alvo:** *RoBERTa-BiLSTM: A Context-Aware Hybrid Model for Sentiment Analysis*.
> 
>**Autores Originais:** Md Mostafizer Rahman, Ariful Islam Shiplu, Yutaka Watanobe, e Md Ashad Alam.
>
> **DOI/Link:** [10.1109/TETCI.2025.3572150](https://ieeexplore.ieee.org/abstract/document/11020722)

---

## 📌 Metodologia e Open Science

Para garantir a transparência e a auditoria de terceiros, este projeto adere estritamente a princípios metodológicos de Ciência Aberta:
* **Ambiente Imutável:** Dependências estritas geridas nativamente via Conda para evitar colisões no hardware (CUDA/Tensor Cores).
* **Determinismo:** Sementes aleatórias (seeds) fixadas em todo o pipeline (`numpy`, `random`, `torch`, e `CUDA`) para garantir que investigadores independentes obtenham os mesmos tensores de peso inicialização.
* **Isolamento de Configuração:** Uso do padrão *12-Factor App* com carregamento de hiperparâmetros via `.env`, separando o código-fonte das variáveis de execução locais.
* **Otimização Mista (AMP):** O código implementa *Automatic Mixed Precision* para aumentar a performance computacional e diminuir o uso de VRAM.

---

## ⚡ Otimizações e Rigor Experimental

* **Aceleração via AMP:** A utilização de precisão mista (FP16/FP32) foi validada através de testes comparativos prévios. Observámos que o impacto na acurácia final é marginal (variação inferior a **-1%**), enquanto o ganho em tempo de treino é superior a 6x em relação à execução em precisão total (FP32), tornando a replicação computacionalmente eficiente e sustentável.
* **Prevenção de Overfitting (Early Stopping):** Integrámos um mecanismo de paragem antecipada (*Early Stopping*) com monitorização da `val_loss`. Esta decisão decorre da observação de que, após a Época 2 ou 3, o modelo tende a sobreajustar-se aos dados de treino. O mecanismo interrompe o ciclo de aprendizagem automaticamente caso não haja melhoria na validação, restaurando os melhores pesos (o "pico de inteligência" da rede), garantindo assim um modelo final mais generalizável e fiável.

---

## ⚙️ Configurações do Ambiente (`.env`)

Todo o comportamento do pipeline, desde o processamento de dados até a arquitetura da rede, é controlado por variáveis de ambiente.

### Reprodutibilidade e Dados
* **`SEED=42`**: Semente matemática global. Garante que os *splits* de dados (90/5/5) e a inicialização dos pesos da rede gerem resultados exatos e reprodutíveis em qualquer máquina.
* **`DATASET_NAME=imdb`**: Define o conjunto de dados a ser processado e avaliado. *(Opções suportadas: `imdb`, `twitter_airline`, `sentiment140`).*
* **`DATASET_MAP_CPU_CORE_NUMBER=8`**: Número de núcleos do processador alocados para paralelizar a tokenização dos dados via API da Hugging Face, acelerando o pré-processamento de bases massivas.
* **`HF_TOKEN=`**: Token de autenticação da Hugging Face. *(Opcional, preencha apenas se for utilizar bases de dados privadas ou com restrição de acesso).*

### Arquitetura do Modelo
* **`MAX_LENGTH=128`**: Número máximo de tokens (palavras/subpalavras) que o modelo RoBERTa irá ler por comentário. Textos maiores são truncados; menores recebem *padding*.
* **`HIDDEN_DIM=256`**: Tamanho das unidades ocultas (neurônios) da camada de recorrência (BiLSTM).
* **`DROPOUT_PROB=0.1`**: Taxa de abandono (10%) aplicada na transição entre o RoBERTa e a BiLSTM para atuar como regularização.

### Hiperparâmetros de Treino
* **`BATCH_SIZE=16`**: Quantidade de comentários processados simultaneamente pela GPU antes da atualização dos pesos.
* **`LEARNING_RATE=1e-5`**: Taxa de aprendizado do otimizador (AdamW). O valor de $1 \times 10^{-5}$ é a zona ideal para *fine-tuning* do RoBERTa segundo o artigo replicado.
* **`USE_AMP=True`**: Ativa o *Automatic Mixed Precision* (FP16). Reduz o uso de VRAM pela metade e acelera significativamente o treino em GPUs modernas (Séries RTX).

### Regularização (Early Stopping dinâmico)
* **`EARLY_STOPPING_MAX_EPOCHS=5`**: Limite rígido máximo de passagens completas pela base de treino.
* **`EARLY_STOPPING_PATIENCE=2`**: Número de épocas que o modelo continuará a treinar sem observar melhorias na perda de validação (*Validation Loss*) antes de ser abortado.
* **`EARLY_STOPPING_DELTA=0.0`**: A melhoria mínima exigida na *Validation Loss* para ser considerada um avanço e resetar o contador de paciência.

---

## 🚀 Instalação e Reprodutibilidade

Para garantir que o cálculo de tensores ocorra num ambiente isolado, é obrigatório o uso do [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda.

**1. Configure as variáveis de ambiente:**
Crie a sua cópia local do arquivo `.env` para poder definir em qual dataset a rede será treinada.

```bash
cp ./src/.env-example .env
```

*(Nota: O arquivo local `.env` é ignorado no `.gitignore` para proteger dados de ambiente).*

**2. Instale o ambiente controlado e ative-o:**
O comando abaixo irá instalar o ecossistema exato, incluindo os binários do CUDA 12.4 de forma isolada do seu sistema operativo nativo.

```bash
conda env create -f environment.yml
conda activate roberta-bilstm-env
```

---

## 📊 Gerando os Gráficos de Distribuição de Dados

Desenvolvemos um script de análise que se conecta à API da Hugging Face, descarrega as bases de dados originais, aplica exatamente a mesma semente matemática (`seed=42`) e a mesma proporção de divisão de dados (`90%` para treino) utilizadas no pipeline principal. Isso garante que o gráfico gerado reflita com precisão absoluta o volume de dados que o modelo processou.

Para visualizar as distribuições, basta executar o seguinte comando na raiz do projeto:

```bash
python -m data_analysis.datasets_analysis 
```

**Resultado esperado:**
O script irá processar as contagens dinamicamente e abrir uma janela interativa do Matplotlib com os gráficos renderizados. A partir dessa janela, você pode inspecionar os dados e utilizar o botão de salvar (ícone de disquete) para exportar a imagem no formato (PNG, PDF, SVG, etc.) e diretório da sua preferência, pronta para ser incluída em artigos e relatórios científicos.

---

## ⚙️ Execução do Pipeline

A arquitetura utiliza o padrão de projeto *Strategy*, permitindo alternar a base de dados em avaliação alterando uma única linha no seu arquivo `.env`.

Abra o seu arquivo `.env` e defina o alvo na variável `DATASET_NAME`:

* `imdb` (Para o dataset de críticas de filmes - IMDb Review)
* `twitter_airline` (Para o dataset de aviação civil - Twitter US Airline Sentiment)
* `sentiment140` (Para o dataset expansivo do Stanford de 1.6M de tweets)

Após definir o ambiente, execute a extração de dados, divisão rigorosa (90/5/5), metra-aprendizagem e validação final com o comando:

```bash
python main.py

```

Os resultados de cada ciclo da experiência serão impressos no terminal e simultaneamente anexados e documentados dentro de `data/experiment_results.json` para histórico e auditoria.

---

## 🔍 Smoke Test: Validação de Integridade

O **Smoke Test** é um teste rápido para garantir que todo o seu pipeline está funcional antes de iniciar o treino real.

### Propósito

* **Testar o Fluxo:** Verifica se o `DataLoader`, o `Modelo` e o `Trainer` se comunicam sem erros.
* **Sem Download:** Usa dados sintéticos (*Dummy Data*), tornando o teste instantâneo e offline.
* **Zero Risco:** Evita perder horas de treino para descobrir um erro simples de configuração.

### Como executar

No seu terminal, dentro da pasta raiz do projeto, execute:

```bash
python -m test.smoke-test
```

### O que acontece?

O sistema treina durante apenas **1 época** com dados de teste. Se ao final ver a mensagem `✅ Smoke test concluído com sucesso!`, o seu sistema está pronto para treinar os datasets reais.

> **Dica:** Use sempre este comando após alterar o código ou o modelo para garantir que não introduziu erros estruturais.

---

## 📊 Métricas Alvo (Ground Truth)

O objetivo principal deste código é reproduzir e balizar a experimentação com base na **Tabela V**, que possuí os melhores resultados documentada no artigo. Ao rodar o pipeline final com a *Learning Rate* em $1\times 10^{-5}$ e *Hidden Units* iguais a 256, os valores-alvo esperados para a validação independente da replicação são:

| Dataset / Base | Acurácia (Acc) | Precisão (Prec) | Recall (Rec) | F1-Score |
| --- | --- | --- | --- | --- |
| **IMDb** | 92.36% | 92.46% | 92.36% | 92.35% |
| **Twitter US Airline** | 80.74% | 80.94% | 80.74% | 80.73% |
| **Sentiment140** | 82.25% | 82.25% | 82.25% | 82.25% |

---

## 📈 Resultados Obtidos

Os resultados experimentais demonstram que a pipeline otimizada não só replicou com sucesso a arquitetura base, como superou as métricas publicadas no artigo original em cenários de grande escala e de desbalanceamento de classes.

### Resumo de Performance

A tabela abaixo apresenta o comparativo direto entre as métricas obtidas nesta replicação independente e os alvos publicados por Rahman *et al.* (2025):

| Dataset | Acurácia<br>(Obtida / Artigo) | Precisão<br>(Obtida / Artigo) | Recall<br>(Obtido / Artigo) | F1-Score<br>(Obtido / Artigo) | Diferença Absoluta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sentiment140** | **89,41%** / 82,25% | **89,42%** / 82,25% | **89,41%** / 82,25% | **89,41%** / 82,25% | **+ 7,16%** 🚀 |
| **Twitter Airline** | **85,66%** / 80,74% | **86,73%** / 80,94% | **85,66%** / 80,74% | **86,03%** / 80,73% | **+ 4,92%** 🚀 |
| **IMDb** | **92,32%** / 92,36% | **92,33%** / 92,46% | **92,32%** / 92,36% | **92,32%** / 92,35% | **- 0,04%** 🎯 |

### Principais Conclusões

1. **Reprodutibilidade Estrita (IMDb):** O modelo atingiu **92,32%** de acurácia contra os 92,36% do artigo original na época 2. Uma variação de apenas -0,04% valida a fidelidade matemática da nossa implementação da arquitetura RoBERTa-BiLSTM.
2. **Mitigação de *Overfitting* (Twitter Airline):** Enquanto os autores originais forçaram o treino por 5 épocas fixas, o nosso mecanismo de *early stopping* interrompeu o treino na **Época 1**. Isto impediu que a rede memorizasse o ruído de uma base pequena e desbalanceada, resultando num salto de **+4,92%** na acurácia de teste.
3. **Ganho por Escala Real (Sentiment140):** Ao utilizar a totalidade do *split* de treino de 90% (1,44 milhões de tweets) protegido por regularização dinâmica, o modelo obteve **89,41%** de acurácia, superando a *baseline* do artigo em **+7,16%**.

### Eficiência e Custo Computacional

Graças à integração de **Automatic Mixed Precision (AMP)** em 16-bits, o pipeline reduziu drasticamente o uso de memória e o tempo de processamento em hardware de consumo local (RTX 4060 Ti 8GB):

* **Twitter US Airline:** Concluído em **04m 43s** (Paragem na Época 1).
* **IMDb Review:** Concluído em **21m 07s** (Paragem na Época 2).
* **Sentiment140:** Concluído em **14h 11m 38s** (Executou as 5 épocas completas, restaurando os melhores pesos da Época 3).

---

## 📜 Licença e Citação

Este repositório é disponibilizado primariamente para investigação e replicação científica sob licença MIT.

Se você fez uso deste código, por favor cite o artigo metodológico original que forneceu os alicerces teóricos para a construção do pipeline:

```bibtex
@article{rahman2025roberta,
  title={RoBERTa-BiLSTM: A Context-Aware Hybrid Model for Sentiment Analysis},
  author={Rahman, Md Mostafizer and Shiplu, Ariful Islam and Watanobe, Yutaka and Alam, Md Ashad},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TETCI.2025.3572150}
}
```