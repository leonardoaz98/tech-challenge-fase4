"""Cria o índice vetorial da base de conhecimento.

Armazenamento: matriz de embeddings em .npy (float32, normalizada), na mesma ordem
das linhas de data/processed/documentos.parquet, que guarda texto e metadados.

Por que não um banco vetorial (Chroma, pgvector, FAISS-HNSW): com ~38 mil vetores,
a busca exata por produto interno em numpy leva poucos milissegundos, não perde
nenhum vizinho (índices aproximados perdem alguns) e combina direto com os filtros
de metadados do pandas. O módulo src/retrieval/indice.py isola essa escolha: trocar
por um banco vetorial muda só aquele arquivo. Ver README §Decisões técnicas.

Uso: python -m src.indexing.indexar
"""

import hashlib
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.indexing.embeddings import codificar_documentos
from src.logger import obter_logger

log = obter_logger(__name__)


def _impressao_digital(docs: pd.DataFrame) -> str:
    """Hash dos ids e textos: o índice só vale para a base que o gerou."""
    h = hashlib.sha256()
    for rid, texto in zip(docs["review_id"], docs["texto"]):
        h.update(f"{rid}\t{texto}\n".encode())
    return h.hexdigest()[:16]


def indexar() -> None:
    docs = pd.read_parquet(config.DOCUMENTOS)
    digital = _impressao_digital(docs)
    if config.EMBEDDINGS.exists() and config.INDICE_META.exists():
        meta = json.loads(config.INDICE_META.read_text())
        if meta.get("base") == digital and meta.get("modelo") == config.MODELO_EMBEDDING:
            log.info("Índice já atualizado (%s)", digital)
            return

    inicio = time.time()
    vetores = codificar_documentos(docs["texto"].tolist())
    config.INDICE.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS, vetores)
    meta = {
        "modelo": config.MODELO_EMBEDDING,
        "base": digital,
        "documentos": int(len(docs)),
        "dimensao": int(vetores.shape[1]),
        "segundos": round(time.time() - inicio, 1),
        "tamanho_mb": round(vetores.nbytes / 1e6, 1),
    }
    config.INDICE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Índice: %s", meta)


if __name__ == "__main__":
    indexar()
