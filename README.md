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

No seu terminal, dentro da pasta raiz do projeto, corra:

```bash
python -m test.smoke-test
```

### O que acontece?

O sistema treina durante apenas **1 época** com dados de teste. Se ao final ver a mensagem `✅ Smoke test concluído com sucesso!`, o seu sistema está pronto para treinar os datasets reais.

> **Dica:** Use sempre este comando após alterar o código ou o modelo para garantir que não introduziu erros estruturais.

---

## 📊 Métricas Alvo (Ground Truth)

O objetivo principal deste código é reproduzir e balizar a experimentação com base na tabela formal documentada no artigo. Ao rodar o pipeline final com a *Learning Rate* em $1\times 10^{-5}$ e *Hidden Units* iguais a 256, os valores-alvo esperados para a validação independente da replicação são:

| Dataset / Base | Acurácia (Acc) | Precisão (Prec) | Recall (Rec) | F1-Score |
| --- | --- | --- | --- | --- |
| **IMDb** | 92.36% | 92.46% | 92.36% | 92.35% |
| **Twitter US Airline** | 80.74% | 80.94% | 80.74% | 80.73% |
| **Sentiment140** | 82.25% | 82.25% | 82.25% | 82.25% |

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