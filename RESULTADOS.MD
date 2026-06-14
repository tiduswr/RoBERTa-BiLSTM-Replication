# Replicação e Otimização do Modelo RoBERTa-BiLSTM para Análise de Sentimentos

## 1. Metodologia: A Arquitetura Híbrida Base

A replicação manteve estritamente o núcleo arquitetural proposto por Rahman *et al.* (2025). O pipeline utiliza o modelo **RoBERTa-base** atrelado ao seu respectivo tokenizador dinâmico para atuar como o extrator contextual primário. Os tensores gerados (com 768 dimensões ocultas) passam por uma camada de *Dropout (0.1)* e são alimentados em uma rede **BiLSTM** com 256 unidades ocultas. A natureza bidirecional da LSTM captura dependências sequenciais de longo alcance no texto, mitigando o esquecimento de contexto. A rede finaliza com camadas densas (Fully Connected) e uma função de ativação Softmax para a classificação de sentimentos (binária ou multiclasse).

## 2. Melhorias Metodológicas e Otimização do Pipeline

Embora a matemática da rede tenha sido preservada, a estratégia de treino foi substancialmente reestruturada para garantir maior generalização e eficiência computacional. As três principais alterações em relação ao estudo original foram:

1. **Rigor na Exposição de Dados (Split 90/5/5 Autêntico):** O artigo original indicou intenção de usar 90% dos dados para treino, mas as métricas de validação indicaram sub-amostragem (especialmente no *Sentiment140*). O nosso pipeline garantiu o carregamento completo via Hugging Face, alimentando a rede com a totalidade dos 90% reais (resultando em mais de 1,44 milhões de instâncias de treino no maior dataset).
2. **Aceleração com Automatic Mixed Precision (AMP):** O treino num hardware de consumo (GPU RTX 4060 Ti de 8GB) num dataset massivo seria impossível devido ao limite de VRAM e processamento em precisão total (FP32). Implementou-se o suporte a AMP (16-bits), o que dividiu o uso de memória pela metade e permitiu treinar 1.44M de tweets em um tempo viável de ~14 horas.
3. **Regularização por Early Stopping Dinâmico:** Os autores originais forçaram um treino estático de 5 épocas, o que naturalmente causou sobreajuste (overfitting) nas épocas finais. Substituímos essa abordagem por um monitoramento contínuo da perda de validação (*Validation Loss*) com paciência de 2 épocas. Isso permitiu congelar e restaurar os pesos no "pico de inteligência" do modelo (Época 1 para o Twitter, Época 2 para o IMDb, e Época 3 para o Sentiment140), o que justificou a superação da *baseline* original em até 7,16%.

## 3. Distribuição e Volume de Dados Treinados

![Distribuição dos Datasets](data/datasets_distribution.png)
*Figura 1: Distribuição do volume exato de instâncias presentes na fatia de 90% de treino processada pelo modelo.*

**Análise da Distribuição:**
A figura acima materializa a densidade estatística com a qual o modelo híbrido foi treinado, refletindo os valores exatos resultantes de um partilhamento matemático dinâmico (`seed=42`). 

* **(a) Twitter US Airline:** Apresenta um forte desbalanceamento natural, focado prioritariamente em reclamações de consumidores. O modelo processou 8.222 instâncias negativas contra 2.803 neutras e 2.151 positivas. Foi justamente este baixo volume de dados aliado ao desbalanceamento que fez com que o nosso *Early Stopping* interrompesse o treino logo na Época 1, impedindo a rede de decorar o ruído.
* **(b) IMDb Review:** Ilustra uma base de dados altamente equilibrada. O modelo treinou com um volume substancial e quase perfeitamente simétrico de avaliações de filmes (22.525 negativas e 22.475 positivas), justificando a rápida convergência e a acurácia superior a 92% na época 2.
* **(c) Sentiment140:** Evidencia o verdadeiro poder de escala do pipeline otimizado com AMP. O modelo ajustou os seus pesos processando um volume colossal de dados balanceados: 719.729 tweets negativos e 720.271 positivos. Essa prova visual da quantidade massiva de conhecimento empírico injetada na rede é o que fundamenta matematicamente o salto brutal de performance obtido nesta replicação em relação aos resultados sub-amostrados do artigo original.

## 4. Resultados e Comparativo de Performance

### Tabela 1: Resumo da Execução e Otimização
*Parâmetros fixos: AMP Ativado, Batch Size = 16, Learning Rate = 1e-05, Hidden Dim = 256.*

| Dataset | Tempo de Treino | Melhor Época (Paciência=2) | Otimização |
| :--- | :--- | :---: | :---: |
| **Sentiment140** | 14h 11m 38s | 3 | AMP (16-bit) |
| **IMDb** | 21m 07s | 2 | AMP (16-bit) |
| **Twitter US Airline** | 04m 43s | 1 | AMP (16-bit) |

---

### Tabela 2: Comparativo de Performance (Nossa Replicação vs. Artigo Original)
*As métricas lado a lado comprovam o ganho de generalização ao utilizar Early Stopping dinâmico e um split de treino otimizado.*

| Dataset | Acurácia<br>(Obtida / Artigo) | Precisão<br>(Obtida / Artigo) | Recall<br>(Obtido / Artigo) | F1-Score<br>(Obtido / Artigo) | Salto de<br>Performance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sentiment140** | **89.41%** / 82.25% | **89.42%** / 82.25% | **89.41%** / 82.25% | **89.41%** / 82.25% | **+ 7.16%** 🚀 |
| **Twitter Airline** | **85.66%** / 80.74% | **86.73%** / 80.94% | **85.66%** / 80.74% | **86.03%** / 80.73% | **+ 4.92%** 🚀 |
| **IMDb** | **92.32%** / 92.36% | **92.33%** / 92.46% | **92.32%** / 92.36% | **92.32%** / 92.35% | **- 0.04%** 🎯 |