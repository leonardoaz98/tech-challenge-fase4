"""Acesso ao índice: documentos, vetores e BM25 carregados uma vez por processo."""

from functools import lru_cache

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src import config
from src.preprocessing.limpeza import tokenizar


class Indice:
    def __init__(self) -> None:
        self.docs = pd.read_parquet(config.DOCUMENTOS)
        self.vetores = np.load(config.EMBEDDINGS)
        if len(self.vetores) != len(self.docs):
            raise RuntimeError("Índice desatualizado: rode python -m src.indexing.indexar")
        self.bm25 = BM25Okapi([tokenizar(t) for t in self.docs["texto"]])
        self.categorias = sorted({c for cs in self.docs["categorias"] for c in cs})

    def busca_densa(self, vetor_consulta: np.ndarray, permitidos: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Similaridade de cosseno exata dentro das posições permitidas pelo filtro."""
        if len(permitidos) == 0:
            return []
        scores = self.vetores[permitidos] @ vetor_consulta
        top = np.argsort(-scores)[:k]
        return [(int(permitidos[i]), float(scores[i])) for i in top]

    def busca_bm25(self, consulta: str, permitidos: np.ndarray, k: int) -> list[tuple[int, float]]:
        tokens = tokenizar(consulta)
        if not tokens or len(permitidos) == 0:
            return []
        scores = np.asarray(self.bm25.get_batch_scores(tokens, permitidos.tolist()))
        top = np.argsort(-scores)[:k]
        return [(int(permitidos[i]), float(scores[i])) for i in top if scores[i] > 0]


@lru_cache(maxsize=1)
def carregar() -> Indice:
    return Indice()
