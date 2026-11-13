"""Valida a classificação de temas contra rótulos de referência.

Passo 1 (sem chave): gera reports/temas_amostra.csv com 300 avaliações sorteadas
(estratificadas por nota) e os temas que as regras atribuíram. O grupo preenche a
coluna `temas_referencia` com os nomes dos temas corretos, separados por vírgula.

Passo 2: calcula precisão e recall por tema contra a coluna preenchida.
Com --llm, a coluna é preenchida pelo Claude (rótulo de referência automático);
a revisão humana continua recomendada (ver README §Limitações).

Uso:
  python -m src.evaluation.avaliar_temas --amostra
  python -m src.evaluation.avaliar_temas --llm       # rotula com o Claude e mede
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
ESQUEMA = {
    "type": "object", "additionalProperties": False, "required": ["itens"],
    "properties": {"itens": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "temas"],
        "properties": {"id": {"type": "string"}, "temas": {"type": "array", "items": {"type": "string", "enum": NOMES}}}}}},
}  # fmt: skip


def rotular_com_llm(amostra: pd.DataFrame, lote: int = 30) -> pd.DataFrame:
    rotulos = {}
    for i in range(0, len(amostra), lote):
        parte = amostra.iloc[i : i + lote]
        texto = "\n".join(f"[{r.review_id}] {r.texto}" for r in parte.itertuples())
        dados, _ = gerar_json(SYSTEM, texto, ESQUEMA, esforco="low")
        rotulos.update({x["id"]: ",".join(x["temas"]) for x in dados["itens"]})
    amostra["temas_referencia"] = amostra["review_id"].map(rotulos).fillna("")
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
