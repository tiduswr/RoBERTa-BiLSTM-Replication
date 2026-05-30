# Como Contribuir para este Projeto

Em primeiro lugar, muito obrigado por dedicar o seu tempo para contribuir!

Este projeto foca-se na replicação do fine-tuning do modelo BERT (especificamente `bert-base-uncased`) para tarefas de Análise de Sentimentos utilizando a framework do Hugging Face e PyTorch. A sua ajuda para melhorar o código, corrigir bugs ou expandir a documentação é muito bem-vinda.

O seguinte documento é um conjunto de diretrizes para contribuir para este repositório no GitHub.

## Índice

- [Como Contribuir para este Projeto](#como-contribuir-para-este-projeto)
  - [Índice](#índice)
  - [Código de Conduta](#código-de-conduta)
  - [Como Posso Contribuir?](#como-posso-contribuir)
    - [Reportar Bugs](#reportar-bugs)
    - [Sugerir Funcionalidades](#sugerir-funcionalidades)
    - [Enviar o seu Primeiro Pull Request (PR)](#enviar-o-seu-primeiro-pull-request-pr)

---

## Código de Conduta

Este projeto e todos os seus participantes estão sujeitos a um Código de Conduta. Ao participar, espera-se que respeite este código. Por favor, reporte qualquer comportamento inaceitável.

## Como Posso Contribuir?

### Reportar Bugs

Esta secção orienta-o na submissão de relatórios de bugs. Siga estas diretrizes para ajudar a equipa a compreender o seu relatório, reproduzir o comportamento e encontrar bugs relacionados.

* **Utilize a pesquisa do GitHub** para verificar se o bug já foi reportado.
* Abra uma *Issue* com um título claro e descritivo.
* **Descreva os passos exatos para reproduzir o problema.** Forneça informações sobre o seu ambiente (ex: Versão do Python, CUDA (ex: 12.1), modelo da GPU (ex: RTX 4060 Ti) e Sistema Operativo).
* Inclua o *stack trace* completo do erro (como erros de `ImportError` ou `TypeError` da biblioteca `transformers`).

### Sugerir Funcionalidades

Esta secção orienta-o na submissão de sugestões de melhorias para o projeto, incluindo novas tarefas (além do SST-2), novos hiperparâmetros ou suporte a outras métricas do GLUE.

* **Utilize a pesquisa do GitHub** para ver se a funcionalidade já foi sugerida.
* Abra uma *Issue* descrevendo qual é a melhoria sugerida.
* Explique *porquê* esta melhoria seria útil (ex: "Adicionar a tarefa QNLI aproxima o projeto ainda mais do artigo original").

### Enviar o seu Primeiro Pull Request (PR)

1. Faça um **Fork** do repositório.
2. Crie um novo *branch* a partir de `master` (`git checkout -b feature/minha-nova-funcionalidade`).
3. Faça as suas alterações no código.
4. Certifique-se de que o código corre sem erros.
5. Faça o commit das suas alterações com mensagens descritivas (`git commit -m "feat: Adiciona suporte à métrica F1"`).
6. Faça *push* para o seu fork (`git push origin feature/minha-nova-funcionalidade`).
7. Submeta um **Pull Request** para revisão.
conda activate bert_env