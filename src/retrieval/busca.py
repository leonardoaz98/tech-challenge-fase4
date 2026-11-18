"""Recuperação híbrida em quatro passos, cada um registrado para demonstração.

1. Filtro de metadados (pré-filtragem).
2. Busca densa (embeddings) + BM25 para cada versão da pergunta.
3. Fusão por Reciprocal Rank Fusion (RRF): soma 1/(k + posição) de cada lista.
   Usa só a posição, então não precisa calibrar escalas diferentes (cosseno × BM25).
4. Reranking com cross-encoder: lê pergunta e avaliação juntas, mais preciso que
   comparar vetores separados, e caro demais para a base inteira; por isso só
   nos 60 melhores candidatos.
5. Seleção: limiar de relevância + MMR (diversidade), sem textos repetidos.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from src import config
from src.indexing.embeddings import codificar_consultas
from src.retrieval.filtros import Filtros, mascara
from src.retrieval.indice import carregar
from src.retrieval.nucleo import nucleo


@dataclass
class Evidencia:
    posicao: int  # linha em documentos.parquet
    review_id: str
    texto: str
    nota: int
    categoria: str
    status_entrega: str
    data_compra: str
    score_rerank: float
    score_rrf: float

    def como_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ResultadoBusca:
    consultas: list[str]
    filtros: Filtros
    universo: int  # documentos que passaram no filtro
    candidatos_denso: int
    candidatos_bm25: int
    candidatos_fundidos: int
    evidencias: list[Evidencia] = field(default_factory=list)
    descartadas_limiar: int = 0
    max_score: float | None = None

    @property
    def suficiente(self) -> bool:
        return len(self.evidencias) >= config.MIN_EVIDENCIAS


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.MODELO_RERANKER, device="cpu")


def rrf(listas: list[list[tuple[int, float]]], k: int = config.RRF_K) -> dict[int, float]:
    fundido: dict[int, float] = {}
    for lista in listas:
        for rank, (pos, _) in enumerate(lista, start=1):
            fundido[pos] = fundido.get(pos, 0.0) + 1.0 / (k + rank)
    return fundido


def mmr(candidatos: list[int], relevancia: dict[int, float], vetores: np.ndarray, n: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: troca um pouco de relevância por diversidade,
    para o contexto não ser 15 variações de "chegou atrasado"."""
    if not candidatos:
        return []
    rel = np.array([relevancia[c] for c in candidatos])
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
    escolhidos: list[int] = []
    restantes = list(range(len(candidatos)))
    V = vetores[candidatos]
    while restantes and len(escolhidos) < n:
        if escolhidos:
            sim = (V[restantes] @ V[escolhidos].T).max(axis=1)
        else:
            sim = np.zeros(len(restantes))
        valor = lam * rel[restantes] - (1 - lam) * sim
        melhor = restantes[int(np.argmax(valor))]
        escolhidos.append(melhor)
        restantes.remove(melhor)
    return [candidatos[i] for i in escolhidos]


def buscar(
    consultas: list[str],
    filtros: Filtros | None = None,
    n_evidencias: int = config.EVIDENCIAS_CONTEXTO,
    usar_bm25: bool = True,
    usar_rerank: bool = True,
    limiar: float | None = config.LIMIAR_RELEVANCIA,
    restringir: set[str] | None = None,
) -> ResultadoBusca:
    """`consultas[0]` é a pergunta original; as demais são reescritas/expansões.
    O limiar de relevância é aplicado ao score contra o núcleo da pergunta original
    (src/retrieval/nucleo.py), o mesmo usado na calibração; a ordem das evidências usa
    o maior score entre o núcleo e as reescritas.
    `restringir`: só busca entre esses review_ids (usado nos exemplos por tema)."""
    filtros = filtros or Filtros()
    idx = carregar()
    m = mascara(idx.docs, filtros)
    if restringir is not None:
        m &= idx.docs["review_id"].isin(restringir)
    permitidos = np.flatnonzero(m.to_numpy())

    # O núcleo também vira consulta: com as reescritas do LLM, as avaliações que falam do
    # assunto com as palavras da pergunta ("tive problemas com a entrega") ficavam de fora.
    alvo = nucleo(consultas[0])
    consultas_busca = list(dict.fromkeys([c for c in consultas if c.strip()] + [alvo]))
    listas_densas, listas_bm25 = [], []
    for vetor, consulta in zip(codificar_consultas(consultas_busca), consultas_busca):
        listas_densas.append(idx.busca_densa(vetor, permitidos, config.CANDIDATOS_DENSO))
        if usar_bm25:
            listas_bm25.append(idx.busca_bm25(consulta, permitidos, config.CANDIDATOS_BM25))
    fundido = rrf(listas_densas + listas_bm25)

    # Textos idênticos ("Produto não chegou") contam uma vez como evidência.
    ordenados, vistos = [], set()
    for pos in sorted(fundido, key=fundido.get, reverse=True):
        chave = idx.docs.at[pos, "texto_busca"]
        if chave not in vistos:
            vistos.add(chave)
            ordenados.append(pos)
    candidatos = ordenados[: config.CANDIDATOS_RERANK]

    resultado = ResultadoBusca(
        consultas=consultas_busca,
        filtros=filtros,
        universo=len(permitidos),
        candidatos_denso=len({p for lst in listas_densas for p, _ in lst}),
        candidatos_bm25=len({p for lst in listas_bm25 for p, _ in lst}),
        candidatos_fundidos=len(fundido),
    )
    if not candidatos:
        return resultado

    if usar_rerank:
        # Dois scores do reranker, com papéis diferentes:
        # - contra o núcleo: decide SE há evidência (é o score do limiar calibrado);
        # - o maior entre núcleo e reescritas: decide QUAIS avaliações entram no contexto.
        # Só com o núcleo ("problemas relacionados à entrega"), relatos concretos
        # ("demorou muito para chegar", sem a palavra "problemas") ficavam abaixo do limiar
        # e negações ("entregue sem problemas") acima. Só com as reescritas, palavras
        # genéricas delas ("entrega", "eficiente") faziam perguntas sem evidência
        # (entrega por drone) passarem do limiar.
        alvos = list(dict.fromkeys([alvo] + [c for c in consultas[1:] if c.strip()]))
        textos = [idx.docs.at[p, "texto"] for p in candidatos]
        pares = [(a, t) for a in alvos for t in textos]
        brutos = np.asarray(_reranker().predict(pares, batch_size=32), dtype=float).reshape(len(alvos), len(candidatos))
        score_nucleo = dict(zip(candidatos, map(float, brutos[0])))
        scores = dict(zip(candidatos, map(float, brutos.max(axis=0))))
        resultado.max_score = max(score_nucleo.values())
    else:
        scores = score_nucleo = {p: fundido[p] for p in candidatos}
        resultado.max_score = max(scores.values())

    if usar_rerank and limiar is not None:
        relevantes = [p for p in candidatos if score_nucleo[p] >= limiar]
        if len(relevantes) >= config.MIN_EVIDENCIAS:
            # Há evidência: a seleção pode usar também quem só pontua alto nas reescritas.
            relevantes = [p for p in candidatos if scores[p] >= limiar]
        resultado.descartadas_limiar = len(candidatos) - len(relevantes)
    else:
        relevantes = candidatos

    selecionados = mmr(relevantes, scores, idx.vetores, n_evidencias, config.MMR_LAMBDA)
    selecionados.sort(key=lambda p: scores[p], reverse=True)
    for p in selecionados:
        d = idx.docs.loc[p]
        resultado.evidencias.append(
            Evidencia(
                posicao=int(p),
                review_id=d["review_id"],
                texto=d["texto"],
                nota=int(d["nota"]),
                categoria=d["categoria"],
                status_entrega=d["status_entrega"],
                data_compra=str(d["data_compra"].date()) if d["data_compra"] is not None else "",
                score_rerank=round(scores[p], 3),
                score_rrf=round(fundido[p], 5),
            )
        )
    return resultado
