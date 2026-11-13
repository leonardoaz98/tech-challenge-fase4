"""Avaliação do RAG contra os critérios do enunciado.

Três medidas:

1. Recusa (roda sem chave): para cada pergunta do conjunto, o sistema respondeu
   quando devia e recusou quando devia?

2. Recuperação (precisa de chave): um juiz LLM dá nota de relevância (0, 1, 2)
   a cada avaliação recuperada. As avaliações de todas as configurações são
   julgadas juntas, uma vez (pooling), e comparadas:
     A densa | B densa + BM25 | C B + reranker | D C + reescrita da pergunta
   Métricas: precisão@10 (relevância >= 1) e nDCG@10.

3. Fidelidade (precisa de chave): o juiz confere se cada insight da resposta é
   sustentado pelas avaliações que ele cita. Junto, a contagem de citações
   inválidas feita pelo código.

O juiz é o mesmo modelo que gera as respostas: isso pode favorecer o próprio
estilo. Por isso uma amostra dos julgamentos vai para revisão humana do grupo
(reports/julgamentos_para_revisar.csv). Ver README §Limitações.

Uso:
  python -m src.evaluation.avaliar            # recusa (com LLM se houver chave)
  python -m src.evaluation.avaliar --juiz     # + recuperação e fidelidade
"""

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from src import config
from src.generation import llm
from src.logger import obter_logger
from src.pipeline import responder
from src.retrieval.analise_pergunta import analisar_com_llm
from src.retrieval.busca import buscar
from src.retrieval.indice import carregar

log = obter_logger(__name__)
PERGUNTAS = Path(__file__).with_name("perguntas.yaml")
JULGAMENTOS = config.RELATORIOS / "julgamentos_relevancia.json"
K = 10


def carregar_perguntas() -> list[dict]:
    return yaml.safe_load(PERGUNTAS.read_text(encoding="utf-8"))["perguntas"]


# ---------------------------------------------------------------- 1. Recusa
def avaliar_recusa(perguntas: list[dict]) -> tuple[dict, list]:
    linhas, resultados = [], []
    for p in perguntas:
        r = responder(p["texto"])
        recusou = r.status in ("fora_do_escopo", "evidencia_insuficiente")
        linhas.append({"id": p["id"], "grupo": p["grupo"], "esperado": p["esperado"], "status": r.status,
                       "acertou": recusou == (p["esperado"] == "recusar"), "max_score": r.busca.max_score if r.busca else None})  # fmt: skip
        resultados.append((p, r))
        log.info("%s %-22s %s", p["id"], r.status, p["texto"][:60])

    def taxa(filtro):
        sel = [l for l in linhas if filtro(l)]
        return round(sum(l["acertou"] for l in sel) / len(sel), 3) if sel else None

    resumo = {
        "modo": "llm" if llm.disponivel() else "regras (sem LLM)",
        "responde_quando_deve": taxa(lambda l: l["esperado"] == "responder"),
        "recusa_fora_do_escopo": taxa(lambda l: l["grupo"] == "fora_escopo"),
        "recusa_evidencia_insuficiente": taxa(lambda l: l["grupo"] == "insuficiente"),
        "por_pergunta": linhas,
    }
    return resumo, resultados


# ------------------------------------------------------------ 2. Recuperação
JUIZ_RELEVANCIA = """Você avalia um sistema de busca sobre avaliações de clientes de e-commerce.
Para a pergunta do gestor, dê a cada avaliação uma nota de relevância:
2 = responde diretamente a pergunta (é exatamente o tipo de relato pedido);
1 = relacionada, ajuda em parte;
0 = não ajuda a responder.
Julgue só pelo texto da avaliação e pelos metadados mostrados. Não seja generoso: um elogio genérico não é relevante para uma pergunta sobre um problema específico."""

ESQUEMA_RELEVANCIA = {
    "type": "object", "additionalProperties": False, "required": ["notas"],
    "properties": {"notas": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "nota"],
        "properties": {"id": {"type": "string"}, "nota": {"type": "integer", "enum": [0, 1, 2]}}}}},
}  # fmt: skip


def _configuracoes(pergunta: str, consultas_llm: list[str], filtros) -> dict[str, list[str]]:
    def ids(**kw):
        return [e.review_id for e in buscar(n_evidencias=K, limiar=None, **kw).evidencias]

    return {
        "A_densa": ids(consultas=[pergunta], usar_bm25=False, usar_rerank=False),
        "B_hibrida": ids(consultas=[pergunta], usar_rerank=False),
        "C_hibrida_rerank": ids(consultas=[pergunta]),
        "D_completa": ids(consultas=consultas_llm, filtros=filtros),
    }


def _julgar(pergunta: str, review_ids: list[str], cache: dict) -> dict:
    faltam = [r for r in review_ids if f"{pergunta}||{r}" not in cache]
    if faltam:
        docs = carregar().docs.set_index("review_id")
        texto = "\n".join(f"[{r}] nota {docs.at[r, 'nota']} | {docs.at[r, 'categoria']} | {docs.at[r, 'status_entrega']}: {docs.at[r, 'texto']}" for r in faltam)
        dados, _ = llm.gerar_json(JUIZ_RELEVANCIA, f"Pergunta: {pergunta}\n\nAvaliações:\n{texto}", ESQUEMA_RELEVANCIA, esforco="low")
        for item in dados["notas"]:
            cache[f"{pergunta}||{item['id']}"] = item["nota"]
    return {r: cache.get(f"{pergunta}||{r}", 0) for r in review_ids}


def _ndcg(notas: list[int]) -> float:
    dcg = sum((2**n - 1) / math.log2(i + 2) for i, n in enumerate(notas))
    ideal = sum((2**n - 1) / math.log2(i + 2) for i, n in enumerate(sorted(notas, reverse=True)))
    return dcg / ideal if ideal else 0.0


def avaliar_recuperacao(perguntas: list[dict]) -> dict:
    cache = json.loads(JULGAMENTOS.read_text()) if JULGAMENTOS.exists() else {}
    por_config: dict[str, list] = {}
    for p in [p for p in perguntas if p["esperado"] == "responder"]:
        analise = analisar_com_llm(p["texto"], carregar().categorias)
        configs = _configuracoes(p["texto"], analise.consultas, analise.filtros)
        pool = sorted({r for ids in configs.values() for r in ids})
        notas = _julgar(p["texto"], pool, cache)
        JULGAMENTOS.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        for nome, ids in configs.items():
            ns = [notas[r] for r in ids]
            por_config.setdefault(nome, []).append(
                {"id": p["id"], "grupo": p["grupo"], "precisao@10": sum(n >= 1 for n in ns) / K, "ndcg@10": _ndcg(ns)}
            )
        log.info("%s julgada (%d avaliações no pool)", p["id"], len(pool))

    def media(lst, chave, grupo=None):
        sel = [x[chave] for x in lst if grupo is None or x["grupo"] == grupo]
        return round(sum(sel) / len(sel), 3) if sel else None

    resumo = {nome: {"precisao@10": media(l, "precisao@10"), "ndcg@10": media(l, "ndcg@10"),
                     "precisao@10_metadados": media(l, "precisao@10", "metadados")} for nome, l in por_config.items()}  # fmt: skip
    _amostra_para_revisao(cache)
    return {"resumo": resumo, "por_pergunta": por_config}


def _amostra_para_revisao(cache: dict, n: int = 60) -> None:
    """Amostra de julgamentos do LLM para o grupo conferir à mão."""
    import random

    docs = carregar().docs.set_index("review_id")
    itens = random.Random(42).sample(sorted(cache.items()), min(n, len(cache)))
    with open(config.RELATORIOS / "julgamentos_para_revisar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pergunta", "review_id", "texto", "nota_llm", "nota_humana"])
        for chave, nota in itens:
            pergunta, rid = chave.split("||")
            w.writerow([pergunta, rid, docs.at[rid, "texto"] if rid in docs.index else "", nota, ""])


# -------------------------------------------------------------- 3. Fidelidade
JUIZ_FIDELIDADE = """Você audita respostas de um sistema RAG. Para cada insight, verifique se as avaliações citadas sustentam o que o insight afirma.
"sustentado": as avaliações citadas dizem isso. "parcial": dizem algo próximo, mas o insight generaliza ou acrescenta. "nao_sustentado": as citações não dizem isso ou não há citação.
Seja rigoroso: números e generalizações ("a maioria", "frequentemente") só são sustentados se vierem da tabela de frequência do contexto."""

ESQUEMA_FIDELIDADE = {
    "type": "object", "additionalProperties": False, "required": ["veredictos"],
    "properties": {"veredictos": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["insight", "veredicto"],
        "properties": {"insight": {"type": "string"},
                       "veredicto": {"type": "string", "enum": ["sustentado", "parcial", "nao_sustentado"]}}}}},
}  # fmt: skip


def avaliar_fidelidade(resultados: list) -> dict:
    linhas = []
    for p, r in resultados:
        if not r.resposta or not r.resposta.insights:
            continue
        insights = "\n".join(f"- {i['insight']} (cita: {', '.join(i['evidencias']) or 'nenhuma'})" for i in r.resposta.insights)
        dados, _ = llm.gerar_json(JUIZ_FIDELIDADE, f"{r.resposta.contexto}\n\nInsights a auditar:\n{insights}", ESQUEMA_FIDELIDADE, esforco="medium")
        v = [x["veredicto"] for x in dados["veredictos"]]
        linhas.append({"id": p["id"], "insights": len(v), "sustentado": v.count("sustentado"), "parcial": v.count("parcial"),
                       "nao_sustentado": v.count("nao_sustentado"), "citacoes_validas": r.resposta.citacoes_validas,
                       "citacoes_invalidas": len(r.resposta.citacoes_invalidas)})  # fmt: skip
    total = sum(l["insights"] for l in linhas) or 1
    return {
        "insights_sustentados": round(sum(l["sustentado"] for l in linhas) / total, 3),
        "insights_parciais": round(sum(l["parcial"] for l in linhas) / total, 3),
        "insights_nao_sustentados": round(sum(l["nao_sustentado"] for l in linhas) / total, 3),
        "citacoes_invalidas": sum(l["citacoes_invalidas"] for l in linhas),
        "citacoes_validas": sum(l["citacoes_validas"] for l in linhas),
        "por_pergunta": linhas,
    }


def _salvar_exemplos(resultados: list) -> None:
    """Perguntas e respostas completas, para o README e para quem não tem chave."""
    saida = [r.como_dict() for _, r in resultados]
    (config.RELATORIOS / "exemplos_respostas.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=str), "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--juiz", action="store_true", help="avalia recuperação e fidelidade com LLM juiz")
    args = ap.parse_args()

    perguntas = carregar_perguntas()
    recusa, resultados = avaliar_recusa(perguntas)
    saida = {"recusa": recusa}
    if llm.disponivel():
        _salvar_exemplos(resultados)
    if args.juiz:
        if not llm.disponivel():
            raise SystemExit("--juiz precisa de ANTHROPIC_API_KEY")
        saida["recuperacao"] = avaliar_recuperacao(perguntas)
        saida["fidelidade"] = avaliar_fidelidade(resultados)
    nome = "avaliacao.json" if llm.disponivel() else "avaliacao_sem_llm.json"
    (config.RELATORIOS / nome).write_text(json.dumps(saida, ensure_ascii=False, indent=2, default=str), "utf-8")
    log.info("Recusa: %s", {k: v for k, v in recusa.items() if k != "por_pergunta"})
    if "recuperacao" in saida:
        log.info("Recuperação: %s", saida["recuperacao"]["resumo"])
        log.info("Fidelidade: %s", {k: v for k, v in saida["fidelidade"].items() if k != "por_pergunta"})


if __name__ == "__main__":
    main()
