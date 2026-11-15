"""Avaliação do RAG contra os critérios do enunciado.

Três medidas:

1. Recusa (roda sem LLM): para cada pergunta do conjunto, o sistema respondeu
   quando devia e recusou quando devia?

2. Recuperação (precisa do Ollama): um juiz LLM dá nota de relevância (0, 1, 2)
   a cada avaliação recuperada. As avaliações de todas as configurações são
   julgadas juntas, uma vez (pooling), e comparadas:
     A densa | B densa + BM25 | C B + reranker | D C + reescrita da pergunta
   Métricas: precisão@10 (relevância >= 1) e nDCG@10.

3. Fidelidade (precisa do Ollama): o juiz confere se cada insight da resposta é
   sustentado pelas avaliações que ele cita. Junto, a contagem de citações
   inválidas feita pelo código.

O juiz é o mesmo modelo que gera as respostas: isso pode favorecer o próprio
estilo. Por isso uma amostra dos julgamentos vai para revisão humana do grupo
(reports/julgamentos_para_revisar.csv). Ver README §Limitações.

Uso:
  python -m src.evaluation.avaliar            # recusa (com LLM se o Ollama estiver no ar)
  python -m src.evaluation.avaliar --juiz     # + recuperação e fidelidade
  python -m src.evaluation.avaliar --etapas fidelidade   # só uma etapa

Com o LLM local, a avaliação completa leva horas. Cada pergunta concluída é salva em
reports/parcial/ (fora do Git); se o processo cair, rodar de novo continua de onde
parou. Cada etapa atualiza só a sua chave em reports/avaliacao.json.
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
PARCIAL = config.RELATORIOS / "parcial"
K = 10


def carregar_perguntas() -> list[dict]:
    return yaml.safe_load(PERGUNTAS.read_text(encoding="utf-8"))["perguntas"]


def _ler(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}


def _gravar(caminho: Path, dados) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=1, default=str), "utf-8")


# ---------------------------------------------------------------- 1. Recusa
def avaliar_recusa(perguntas: list[dict], usar_llm: bool) -> tuple[dict, list]:
    """Retorna o resumo e a lista (pergunta, resultado como dict). Retoma do parcial."""
    parcial = PARCIAL / f"recusa_{'llm' if usar_llm else 'regras'}.json"
    feito = _ler(parcial)
    linhas, resultados = [], []
    for p in perguntas:
        if p["id"] in feito and feito[p["id"]]["texto"] == p["texto"]:
            item = feito[p["id"]]
        else:
            r = responder(p["texto"], usar_llm=usar_llm)
            recusou = r.status in ("fora_do_escopo", "evidencia_insuficiente")
            linha = {"id": p["id"], "grupo": p["grupo"], "esperado": p["esperado"], "status": r.status,
                     "acertou": recusou == (p["esperado"] == "recusar"), "max_score": r.busca.max_score if r.busca else None}  # fmt: skip
            item = feito[p["id"]] = {"texto": p["texto"], "linha": linha, "resultado": r.como_dict()}
            _gravar(parcial, feito)
            log.info("%s %-22s %s", p["id"], r.status, p["texto"][:60])
        linhas.append(item["linha"])
        resultados.append((p, item["resultado"]))

    def taxa(filtro):
        sel = [l for l in linhas if filtro(l)]
        return round(sum(l["acertou"] for l in sel) / len(sel), 3) if sel else None

    resumo = {
        "modo": f"llm ({config.MODELO_LLM})" if usar_llm else "regras (sem LLM)",
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



def _esquema_relevancia(n: int) -> dict:
    """Uma nota por avaliação, na ordem em que foram mostradas. Posicional e com
    tamanho fixo: um modelo de 7B erra ao copiar review_ids de 32 caracteres."""
    nota = {"type": "integer", "enum": [0, 1, 2]}
    return {"type": "object", "additionalProperties": False, "required": ["notas"],
            "properties": {"notas": {"type": "array", "items": nota, "minItems": n, "maxItems": n}}}  # fmt: skip


def _configuracoes(pergunta: str, consultas_llm: list[str], filtros) -> dict[str, list[str]]:
    def ids(**kw):
        return [e.review_id for e in buscar(n_evidencias=K, limiar=None, **kw).evidencias]

    return {
        "A_densa": ids(consultas=[pergunta], usar_bm25=False, usar_rerank=False),
        "B_hibrida": ids(consultas=[pergunta], usar_rerank=False),
        "C_hibrida_rerank": ids(consultas=[pergunta]),
        "D_completa": ids(consultas=consultas_llm, filtros=filtros),
    }


def _julgar(pergunta: str, review_ids: list[str], cache: dict, lote: int = 10) -> dict:
    faltam = [r for r in review_ids if f"{pergunta}||{r}" not in cache]
    docs = carregar().docs.set_index("review_id")
    for i in range(0, len(faltam), lote):
        parte = faltam[i : i + lote]
        texto = "\n".join(
            f"{j}. nota {docs.at[r, 'nota']} | {docs.at[r, 'categoria']} | {docs.at[r, 'status_entrega']}: {docs.at[r, 'texto']}"
            for j, r in enumerate(parte, 1)
        )
        dados, _ = llm.gerar_json(
            JUIZ_RELEVANCIA, f"Pergunta: {pergunta}\n\nAvaliações ({len(parte)}):\n{texto}\n\nDevolva {len(parte)} notas, na ordem.",
            _esquema_relevancia(len(parte)), max_tokens=200,
        )
        for r, nota in zip(parte, dados["notas"]):
            cache[f"{pergunta}||{r}"] = nota
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
    """Recebe (pergunta, resultado como dict) da etapa de recusa. Retoma do parcial."""
    parcial = PARCIAL / "fidelidade.json"
    feito = _ler(parcial)
    linhas = []
    for p, r in resultados:
        resp = r.get("resposta")
        if not resp or not resp.get("insights"):
            continue
        insights = "\n".join(f"- {i['insight']} (cita: {', '.join(i['evidencias']) or 'nenhuma'})" for i in resp["insights"])
        if feito.get(p["id"], {}).get("insights_texto") != insights:
            dados, _ = llm.gerar_json(JUIZ_FIDELIDADE, f"{resp['contexto']}\n\nInsights a auditar:\n{insights}", ESQUEMA_FIDELIDADE)
            v = [x["veredicto"] for x in dados["veredictos"]]
            feito[p["id"]] = {"insights_texto": insights, "linha": {
                "id": p["id"], "insights": len(v), "sustentado": v.count("sustentado"), "parcial": v.count("parcial"),
                "nao_sustentado": v.count("nao_sustentado"), "citacoes_validas": resp["citacoes_validas"],
                "citacoes_invalidas": len(resp["citacoes_invalidas"])}}  # fmt: skip
            _gravar(parcial, feito)
            log.info("%s fidelidade julgada", p["id"])
        linhas.append(feito[p["id"]]["linha"])
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
    """Perguntas e respostas completas, para o README e para quem não roda o LLM."""
    _gravar(config.RELATORIOS / "exemplos_respostas.json", [r for _, r in resultados])


ETAPAS = ("recusa", "recuperacao", "fidelidade")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--juiz", action="store_true", help="avalia também recuperação e fidelidade com LLM juiz")
    ap.add_argument("--etapas", help=f"etapas separadas por vírgula, entre {', '.join(ETAPAS)}")
    ap.add_argument("--sem-llm", action="store_true", help="mede o modo sem LLM mesmo com o Ollama no ar")
    args = ap.parse_args()

    etapas = args.etapas.split(",") if args.etapas else (list(ETAPAS) if args.juiz else ["recusa"])
    if invalidas := set(etapas) - set(ETAPAS):
        raise SystemExit(f"etapas desconhecidas: {invalidas}")
    usar_llm = llm.disponivel() and not args.sem_llm
    if set(etapas) & {"recuperacao", "fidelidade"} and not usar_llm:
        raise SystemExit("recuperação e fidelidade precisam do Ollama rodando com o modelo baixado (ver README, seção 13)")

    perguntas = carregar_perguntas()
    arquivo = config.RELATORIOS / ("avaliacao.json" if usar_llm else "avaliacao_sem_llm.json")
    saida = _ler(arquivo)

    def salvar(chave, valor):
        saida[chave] = valor
        arquivo.write_text(json.dumps(saida, ensure_ascii=False, indent=2, default=str), "utf-8")

    if "recusa" in etapas or "fidelidade" in etapas:
        # A fidelidade audita as respostas da etapa de recusa; vêm do parcial se já existirem.
        recusa, resultados = avaliar_recusa(perguntas, usar_llm)
        salvar("recusa", recusa)
        if usar_llm:
            _salvar_exemplos(resultados)
        log.info("Recusa: %s", {k: v for k, v in recusa.items() if k != "por_pergunta"})
    if "recuperacao" in etapas:
        salvar("recuperacao", avaliar_recuperacao(perguntas))
        log.info("Recuperação: %s", saida["recuperacao"]["resumo"])
    if "fidelidade" in etapas:
        salvar("fidelidade", avaliar_fidelidade(resultados))
        log.info("Fidelidade: %s", {k: v for k, v in saida["fidelidade"].items() if k != "por_pergunta"})


if __name__ == "__main__":
    main()
