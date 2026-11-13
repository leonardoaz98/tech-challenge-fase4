"""Calibra o limiar de relevância do reranker (quando dizer "não sei").

O limiar decide se uma avaliação conta como evidência. Com menos de
config.MIN_EVIDENCIAS acima dele, o sistema recusa. A calibração usa um conjunto
PRÓPRIO de 20 perguntas, diferente do conjunto de avaliação, para que a medida de
recusa em src/evaluation/avaliar.py não seja feita com as mesmas perguntas que
escolheram o limiar.

Roda sem chave de API (só busca + reranker).
Uso: python -m src.evaluation.calibrar_limiar
"""

import json

import numpy as np

from src import config
from src.logger import obter_logger
from src.retrieval.analise_pergunta import analisar_por_regras
from src.retrieval.busca import buscar

log = obter_logger(__name__)

CALIBRACAO = {
    "responder": [
        "O que os clientes falam sobre o tamanho das roupas?",
        "Há reclamações sobre produtos que vieram sem manual de instruções?",
        "O que os clientes dizem sobre a cor do produto?",
        "Como os clientes descrevem entregas que chegaram antes do prazo?",
        "O que os clientes relatam sobre brinquedos?",
        "Há elogios ao vendedor?",
        "O que os clientes relatam sobre produtos com voltagem errada?",
        "O que os clientes dizem sobre perfumes?",
        "Há reclamações sobre cartuchos de impressora?",
        "O que os clientes falam sobre o prazo de entrega estimado?",
    ],
    "recusar": [
        "Qual é a taxa de juros do Banco Central?",
        "Quem ganhou a Copa do Mundo de 2002?",
        "Como faço um bolo de chocolate?",
        "Qual o salário médio de um entregador?",
        "Traduza bom dia para o inglês.",
        "Os clientes elogiam a loja física em Paris?",
        "O que os clientes acham do serviço de streaming da loja?",
        "Como os clientes avaliam o test drive de carros elétricos?",
        "Há relatos sobre o seguro viagem vendido pela loja?",
        "O que os clientes acham do restaurante da loja?",
    ],
}


def _scores(pergunta: str) -> list[float]:
    # Mesmas consultas do modo sem LLM; o reranker usa o núcleo da pergunta nos dois modos.
    r = buscar(analisar_por_regras(pergunta).consultas, limiar=None, n_evidencias=config.CANDIDATOS_RERANK)
    return sorted((e.score_rerank for e in r.evidencias), reverse=True)


def calibrar() -> dict:
    dados = {rotulo: {p: _scores(p) for p in ps} for rotulo, ps in CALIBRACAO.items()}
    grade = np.round(np.arange(-6, 8.01, 0.25), 2)
    linhas = []
    for t in grade:
        def responde(s):
            return sum(x >= t for x in s) >= config.MIN_EVIDENCIAS
        acerto_resp = np.mean([responde(s) for s in dados["responder"].values()])
        acerto_rec = np.mean([not responde(s) for s in dados["recusar"].values()])
        linhas.append({"limiar": float(t), "responde_quando_deve": float(acerto_resp),
                       "recusa_quando_deve": float(acerto_rec), "acuracia_balanceada": float((acerto_resp + acerto_rec) / 2)})  # fmt: skip

    melhor = max(r["acuracia_balanceada"] for r in linhas)
    empatados = [r for r in linhas if r["acuracia_balanceada"] == melhor]
    # Entre limiares empatados, o do meio da faixa: mais margem para os dois lados.
    escolhido = empatados[len(empatados) // 2]
    k = config.MIN_EVIDENCIAS - 1
    saida = {
        "regra": f"responde se pelo menos {config.MIN_EVIDENCIAS} avaliações têm score do reranker >= limiar",
        "limiar_escolhido": escolhido["limiar"],
        "faixa_empatada": [empatados[0]["limiar"], empatados[-1]["limiar"]],
        "acuracia_balanceada": escolhido["acuracia_balanceada"],
        "score_da_3a_melhor_avaliacao": {
            rot: {p: round(s[k], 2) if len(s) > k else None for p, s in ps.items()} for rot, ps in dados.items()
        },
        "grade": linhas,
    }
    (config.RELATORIOS / "calibracao_limiar.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), "utf-8")
    _grafico(dados, escolhido["limiar"], k)
    log.info("Limiar escolhido: %s (acurácia balanceada %.2f, faixa %s)", escolhido["limiar"],
             escolhido["acuracia_balanceada"], saida["faixa_empatada"])  # fmt: skip
    return saida


def _grafico(dados: dict, limiar: float, k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    cores = {"responder": "#2a7ab9", "recusar": "#c0392b"}
    for i, (rot, ps) in enumerate(dados.items()):
        vals = [s[k] for s in ps.values() if len(s) > k]
        ax.scatter(vals, np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)),
                   color=cores[rot], s=40, alpha=0.8, label=f"deveria {rot}")  # fmt: skip
    ax.axvline(limiar, color="#333", ls="--", lw=1)
    ax.text(limiar, 1.45, f" limiar = {limiar}", fontsize=9, va="top")
    ax.set_yticks([0, 1], ["deveria responder", "deveria recusar"])
    ax.set_xlabel(f"score do reranker da {k + 1}ª avaliação mais relevante")
    ax.set_title("Calibração do limiar de relevância (20 perguntas fora do conjunto de avaliação)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    config.IMAGENS.mkdir(exist_ok=True)
    fig.savefig(config.IMAGENS / "calibracao_limiar.png", dpi=150)


if __name__ == "__main__":
    calibrar()
