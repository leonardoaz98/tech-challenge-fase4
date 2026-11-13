"""Pipeline RAG de ponta a ponta. Cada etapa fica registrada no resultado.

pergunta
 -> análise (escopo, tipo, filtros, reescritas)          src/retrieval/analise_pergunta.py
 -> [fora do escopo] recusa explicada
 -> recuperação híbrida + reranking + limiar + MMR       src/retrieval/busca.py
 -> [perguntas de frequência] contagem de temas na base   src/temas/temas.py
 -> [evidência insuficiente] recusa explicada, sem chamar o LLM
    (frequência: universo filtrado pequeno; demais: menos de 3 avaliações acima do limiar)
 -> contexto + geração com citações + verificação         src/generation/resposta.py

Uso: python -m src.pipeline "Quais são os principais problemas relatados?"
"""

import json
import sys
import time
from dataclasses import asdict, dataclass, field

from src import config
from src.generation import llm
from src.generation.resposta import Resposta, gerar
from src.retrieval.analise_pergunta import Analise, analisar_com_llm, analisar_por_regras
from src.retrieval.busca import ResultadoBusca, buscar
from src.retrieval.filtros import Filtros
from src.retrieval.indice import carregar
from src.temas import temas

MSG_FORA_ESCOPO = "Esta pergunta está fora do que a base de avaliações de clientes consegue responder. {motivo}"
MSG_INSUFICIENTE = (
    "Não encontrei avaliações suficientes para responder com segurança "
    "({n} avaliações relevantes; o mínimo é {minimo}). Tente ampliar os filtros ou reformular a pergunta."
)
MSG_UNIVERSO = (
    "Os filtros deixaram só {n} avaliações com texto (mínimo {minimo} para uma contagem de temas "
    "fazer sentido). Amplie o período, a categoria ou a faixa de nota."
)


@dataclass
class ResultadoRAG:
    pergunta: str
    status: str  # respondida | fora_do_escopo | evidencia_insuficiente | so_recuperacao
    mensagem: str
    analise: Analise
    busca: ResultadoBusca | None = None
    frequencias: dict | None = None
    resposta: Resposta | None = None
    tempos: dict = field(default_factory=dict)

    def como_dict(self) -> dict:
        d = asdict(self)
        if self.busca:
            d["busca"]["filtros"] = asdict(self.busca.filtros)
        return d


def _evidencias_por_tema(frequencias: dict, filtros: Filtros, n_temas: int = 5, por_tema: int = 3):
    """Para perguntas de frequência: exemplos dos temas mais comuns, além da busca geral.
    Assim cada número da tabela tem avaliações que o ilustram."""
    import pandas as pd

    marcados = pd.read_parquet(config.TEMAS)
    exemplos = []
    for t in [t for t in frequencias["temas"] if t["avaliacoes"] > 0][:n_temas]:
        ids_tema = set(marcados.loc[marcados[t["tema"]], "review_id"])
        # Sem reranker aqui: pertencer ao tema já garante a relevância, e o reranker
        # em 5 temas somaria ~10 s por pergunta em CPU.
        r = buscar([t["rotulo"]], filtros, n_evidencias=por_tema, usar_bm25=False, usar_rerank=False, restringir=ids_tema)
        exemplos.extend(r.evidencias)
    return exemplos


def responder(pergunta: str, filtros_interface: Filtros | None = None, usar_llm: bool | None = None) -> ResultadoRAG:
    tempos, t0 = {}, time.time()
    usar_llm = llm.disponivel() if usar_llm is None else usar_llm
    idx = carregar()
    tempos["carregar_indice"] = round(time.time() - t0, 2)

    # 1. Processamento da pergunta
    t = time.time()
    if usar_llm:
        analise = analisar_com_llm(pergunta, idx.categorias)
        if filtros_interface and not filtros_interface.vazio():
            analise.filtros = filtros_interface  # o que o usuário marcou na tela tem prioridade
    else:
        analise = analisar_por_regras(pergunta, filtros_interface)
    tempos["analise"] = round(time.time() - t, 2)

    if not analise.no_escopo:
        return ResultadoRAG(pergunta, "fora_do_escopo", MSG_FORA_ESCOPO.format(motivo=analise.motivo), analise, tempos=tempos)

    # 2. Recuperação
    t = time.time()
    busca = buscar(analise.consultas, analise.filtros)
    frequencias = None
    evidencias = list(busca.evidencias)
    n_relevantes, suficiente = len(busca.evidencias), busca.suficiente
    if analise.tipo == "frequencia":
        # "Quais os principais problemas?" não tem um assunto para o reranker medir:
        # a resposta vem da contagem de temas no universo filtrado, e as evidências
        # são exemplos de cada tema. Suficiente = universo grande o bastante para um
        # percentual fazer sentido e pelo menos MIN_EVIDENCIAS exemplos.
        tipo = None if analise.tipo_tema == "todos" else analise.tipo_tema
        frequencias = temas.contar(analise.filtros, tipo=tipo)
        vistos = {e.review_id for e in evidencias}
        exemplos = [e for e in _evidencias_por_tema(frequencias, analise.filtros) if e.review_id not in vistos]
        evidencias += exemplos
        n_relevantes = len(evidencias)
        suficiente = frequencias["universo"] >= config.MIN_UNIVERSO_FREQUENCIA and n_relevantes >= config.MIN_EVIDENCIAS
    tempos["recuperacao"] = round(time.time() - t, 2)

    if not suficiente:
        msg = MSG_INSUFICIENTE.format(n=n_relevantes, minimo=config.MIN_EVIDENCIAS)
        if frequencias and frequencias["universo"] < config.MIN_UNIVERSO_FREQUENCIA:
            msg = MSG_UNIVERSO.format(n=frequencias["universo"], minimo=config.MIN_UNIVERSO_FREQUENCIA)
        return ResultadoRAG(pergunta, "evidencia_insuficiente", msg, analise, busca, frequencias, tempos=tempos)

    busca.evidencias = evidencias
    if not usar_llm:
        msg = "Modo só recuperação (sem chave da API): abaixo, as avaliações mais relevantes para a pergunta."
        return ResultadoRAG(pergunta, "so_recuperacao", msg, analise, busca, frequencias, tempos=tempos)

    # 3. Geração
    t = time.time()
    resposta = gerar(pergunta, analise.filtros.descrever(), evidencias, frequencias)
    tempos["geracao"] = round(time.time() - t, 2)
    tempos["total"] = round(time.time() - t0, 2)
    status = "respondida" if resposta.suficiente else "evidencia_insuficiente"
    return ResultadoRAG(pergunta, status, resposta.texto, analise, busca, frequencias, resposta, tempos)


def imprimir(r: ResultadoRAG) -> None:
    a = r.analise
    print(f"\n=== {r.pergunta}")
    print(f"[1] Análise ({a.origem}): escopo={a.no_escopo} tipo={a.tipo} tema={a.tipo_tema} filtros={a.filtros.descrever()}")
    for c in a.consultas[1:]:
        print(f"    reescrita: {c}")
    if r.busca:
        b = r.busca
        print(f"[2] Busca: universo={b.universo} denso={b.candidatos_denso} bm25={b.candidatos_bm25} "
              f"fundidos={b.candidatos_fundidos} abaixo_limiar={b.descartadas_limiar} max_score={b.max_score}")  # fmt: skip
    if r.frequencias:
        print("[3] Frequência de temas:", ", ".join(f"{t['rotulo']} {t['percentual']}%" for t in r.frequencias["temas"][:5]))
    print(f"\n[{r.status}] {r.mensagem}\n")
    if r.busca:
        for i, e in enumerate(r.busca.evidencias, 1):
            print(f"  [R{i}] ({e.score_rerank:+.2f}) nota {e.nota} | {e.categoria} | {e.status_entrega}: {e.texto[:150]}")
    if r.resposta:
        print(f"\ncitações válidas={r.resposta.citacoes_validas} inválidas={r.resposta.citacoes_invalidas} lacunas: {r.resposta.lacunas}")
    print("tempos:", json.dumps(r.tempos))


if __name__ == "__main__":
    imprimir(responder(" ".join(sys.argv[1:]) or "Quais são os principais problemas relatados pelos clientes?"))
