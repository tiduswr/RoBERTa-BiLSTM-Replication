# Contribuindo para a Replicação RoBERTa-BiLSTM

Obrigado pelo seu interesse em contribuir! Como este é um projeto focado em **reprodutibilidade científica**, pedimos que siga estas diretrizes para garantir que as alterações não comprometam a integridade dos resultados experimentais.

## 🚀 Como começar
1. **Fork** o repositório.
2. Crie uma branch para a sua alteração: `git checkout -b feature/nome-da-melhoria`.
3. Verifique se o ambiente está configurado conforme o `environment.yml`.

## 🧪 Regras de Ouro (Ciência Aberta)
* **Não altere o pipeline central sem teste:** Qualquer alteração no `src/trainer.py` ou `src/model.py` deve ser validada primeiro com o **Smoke Test**:
```bash
  python -m test.smoke-test
```

* **Mantenha o Determinismo:** Não remova a fixação de `seeds` ou a estrutura de carregamento de hiperparâmetros do `.env`.
* **Datasets:** Não altere a lógica de `train_test_split` (90/5/5) para não invalidar a comparabilidade com o artigo original.

## 🛠️ Desenvolvimento

* **Adicionar um novo Dataset:**
1. Crie uma nova classe de estratégia em `src/dataset_strategies.py` implementando o protocolo `DatasetStrategy`.
2. Adicione-a ao dicionário no `main.py` e ao `config.py` (para as métricas alvo).


* **Pull Requests:**
* Descreva claramente o objetivo da alteração.
* Se a mudança afetar o tempo de treino ou a precisão, anexe um log comparativo (Obtido vs. Alvo).

## 🐛 Reportar Problemas

Ao reportar um problema, forneça:

1. Logs do terminal.
2. Versão da GPU e CUDA (`nvidia-smi`).
3. O comando utilizado para rodar o pipeline.

---

*Ao submeter um PR, você concorda que o seu código será verificado para garantir que a replicação permanece fiel à proposta científica original.*
