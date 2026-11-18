"""Valida a classificação de temas contra rótulos de referência.

Passo 1 (sem LLM): gera reports/temas_amostra.csv com 300 avaliações sorteadas
(estratificadas por nota) e os temas que as regras atribuíram. O grupo preenche a
coluna `temas_referencia` com os nomes dos temas corretos, separados por vírgula.

Passo 2: calcula precisão e recall por tema contra a coluna preenchida.
Com --llm, a coluna é preenchida pelo LLM local (rótulo de referência automático);
a revisão humana continua recomendada (ver README §Limitações).

Uso:
  python -m src.evaluation.avaliar_temas --amostra
  python -m src.evaluation.avaliar_temas --llm       # rotula com o LLM local e mede
  python -m src.evaluation.avaliar_temas             # mede com a coluna já preenchida
"""

import argparse
import json

import pandas as pd

from src import config
from src.generation.llm import gerar_json
from src.logger import obter_logger
from src.temas.temas import TEMAS

log = obter_logger(__name__)
AMOSTRA = config.RELATORIOS / "temas_amostra.csv"
NOMES = [t.nome for t in TEMAS]


def gerar_amostra(n: int = 300) -> pd.DataFrame:
    docs = pd.read_parquet(config.DOCUMENTOS, columns=["review_id", "texto", "nota"])
    marcados = pd.read_parquet(config.TEMAS)
    base = docs.merge(marcados, on="review_id")
    amostra = base.groupby("nota").sample(n // 5, random_state=42)
    amostra["temas_regras"] = amostra[NOMES].apply(lambda r: ",".join(n for n in NOMES if r[n]), axis=1)
    amostra["temas_referencia"] = ""
    amostra[["review_id", "nota", "texto", "temas_regras", "temas_referencia"]].to_csv(AMOSTRA, index=False)
    log.info("Amostra para rotular: %s", AMOSTRA)
    return amostra


SYSTEM = "Você classifica avaliações de clientes de e-commerce em temas. Marque todos os temas que a avaliação menciona de forma explícita, e só esses.\n\nTemas:\n" + "\n".join(
    f"- {t.nome}: {t.rotulo} ({t.tipo})" for t in TEMAS
)
# Um booleano por tema, e não uma lista de temas: com lista, o modelo de 7B às vezes
# entra em laço repetindo os mesmos itens até estourar o limite de tokens (e os
# rótulos salvos antes do laço ser detectado vinham contaminados). Com campos fixos
# a saída tem sempre o mesmo tamanho e o modelo decide tema a tema.
ESQUEMA = {
    "type": "object", "additionalProperties": False, "required": NOMES,
    "properties": {n: {"type": "boolean"} for n in NOMES},
}  # fmt: skip


def rotular_com_llm(amostra: pd.DataFrame) -> pd.DataFrame:
    """Uma avaliação por chamada: com 14 temas possíveis, lotes grandes confundem um
    modelo de 7B. ~300 chamadas curtas, alguns minutos com GPU. Cada rótulo é salvo em
    reports/parcial/ e, se o processo cair, a próxima execução continua de onde parou."""
    parcial = config.RELATORIOS / "parcial" / "temas_rotulos_v2.json"
    feito = json.loads(parcial.read_text(encoding="utf-8")) if parcial.exists() else {}
    parcial.parent.mkdir(parents=True, exist_ok=True)
    rotulos = []
    for i, r in enumerate(amostra.itertuples(), 1):
        if r.review_id not in feito:
            dados, _ = gerar_json(SYSTEM, f"Avaliação: {r.texto}", ESQUEMA, max_tokens=400)
            feito[r.review_id] = ",".join(n for n in NOMES if dados[n])
            parcial.write_text(json.dumps(feito, ensure_ascii=False), "utf-8")
        rotulos.append(feito[r.review_id])
        if i % 50 == 0:
            log.info("%d/%d rotuladas", i, len(amostra))
    amostra["temas_referencia"] = rotulos
    amostra.to_csv(AMOSTRA, index=False)
    return amostra


def medir(amostra: pd.DataFrame) -> dict:
    def conjunto(s):
        return {x for x in str(s).split(",") if x and x != "nan"}

    pred = amostra["temas_regras"].map(conjunto)
    ref = amostra["temas_referencia"].map(conjunto)
    por_tema = {}
    for n in NOMES:
        vp = sum(n in p and n in r for p, r in zip(pred, ref))
        fp = sum(n in p and n not in r for p, r in zip(pred, ref))
        fn = sum(n not in p and n in r for p, r in zip(pred, ref))
        por_tema[n] = {"precisao": round(vp / (vp + fp), 2) if vp + fp else None,
                       "recall": round(vp / (vp + fn), 2) if vp + fn else None, "suporte": vp + fn}  # fmt: skip
    saida = {"amostra": len(amostra), "referencia": "ver coluna temas_referencia em temas_amostra.csv", "por_tema": por_tema}
    (config.RELATORIOS / "temas_validacao.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), "utf-8")
    log.info("Validação de temas: %s", por_tema)
    return saida


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostra", action="store_true")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    if a.amostra:
        gerar_amostra()
    else:
        amostra = pd.read_csv(AMOSTRA) if AMOSTRA.exists() else gerar_amostra()
        if a.llm:
            amostra = rotular_com_llm(amostra)
        medir(amostra)
