# Como trabalhar neste repositório

Quem escreve o código abre o pull request; o grupo revisa antes do merge. Nada entra na `main` sem branch, pull request e CI verde.

## 1. Uma branch por mudança

    git checkout main && git pull
    git checkout -b tipo/descricao-curta

| Prefixo | Quando usar | Exemplo |
|---|---|---|
| `feat/` | Etapa nova do pipeline, análise ou dado novo | `feat/recuperacao-hibrida` |
| `fix/` | Correção de algo que estava errado | `fix/negacao-temas` |
| `docs/` | README, documentação, notebooks | `docs/readme` |
| `ci/` | Teste automático | `ci/github-actions` |
| `refactor/` | Reescrita sem mudar resultado | `refactor/indice` |

## 2. Commits que explicam o porquê

    tipo: o que mudou, em uma linha

    Por que mudou e o que isso muda no resultado. O "o quê" já está no diff.

## 3. Antes de abrir o pull request

    python -m pytest -q tests
    python -m src.run_pipeline          # do zero; com ANTHROPIC_API_KEY roda também a avaliação com LLM
    python -m src.checar_resultados     # funil e temas iguais à referência?

- Se o funil ou os temas mudaram **de propósito**, grave a referência nova (`python -m src.checar_resultados --gravar`) num commit próprio, explicando o motivo.
- Mudou algo na busca, no prompt ou nos limiares? Rode `python -m src.evaluation.avaliar --juiz` e coloque no PR o antes e depois das métricas.
- Toda decisão técnica nova entra na tabela de decisões do README, com motivo e evidência.

## 4. Pull request

- Preencha o modelo que o GitHub abre (`.github/pull_request_template.md`).
- O GitHub Actions roda testes, base de conhecimento e temas numa máquina limpa. Com o CI vermelho, o PR não entra.
- Revisão: outra pessoa do grupo lê o diff e a descrição antes do merge. Merge com `--merge`, para o histórico mostrar de qual branch veio cada mudança.
