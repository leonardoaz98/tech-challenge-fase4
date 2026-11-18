"""Modelo de embeddings.

Escolha: intfloat/multilingual-e5-base.
- Multilíngue, treinado com português; os comentários são informais e curtos.
- Roda em CPU, sem chave de API: qualquer pessoa reproduz o índice.
- Os modelos E5 foram treinados com prefixos: "query: " na pergunta e
  "passage: " no documento. Sem eles a qualidade da busca cai.
- Vetores normalizados: produto interno = similaridade de cosseno.
"""

from functools import lru_cache

import numpy as np

from src import config


@lru_cache(maxsize=1)
def _modelo():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.MODELO_EMBEDDING, device="cpu")


def codificar_documentos(textos: list[str], mostrar_progresso: bool = True) -> np.ndarray:
    vetores = _modelo().encode(
        [f"passage: {t}" for t in textos],
        batch_size=config.LOTE_EMBEDDING,
        normalize_embeddings=True,
        show_progress_bar=mostrar_progresso,
    )
    return vetores.astype(np.float32)


def codificar_consultas(consultas: list[str]) -> np.ndarray:
    vetores = _modelo().encode(
        [f"query: {c}" for c in consultas], normalize_embeddings=True, show_progress_bar=False
    )
    return vetores.astype(np.float32)
