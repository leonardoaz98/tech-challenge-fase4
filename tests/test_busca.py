"""Fusão de rankings e filtros de metadados. Não depende de modelos nem da base."""

import pandas as pd

from src.retrieval.busca import rrf
from src.retrieval.filtros import Filtros, mascara


def test_rrf_premia_quem_aparece_nas_duas_listas():
    fundido = rrf([[(1, 0.9), (2, 0.8)], [(2, 10.0), (3, 5.0)]])
    assert max(fundido, key=fundido.get) == 2


def test_filtros():
    docs = pd.DataFrame({
        "nota": [1, 5, 2], "data_compra": pd.to_datetime(["2017-11-24", "2018-01-10", "2017-11-30"]),
        "categorias": [["cama mesa banho"], ["relogios presentes"], ["cama mesa banho"]],
        "status_entrega": ["atrasado", "no_prazo", "nao_entregue"], "uf_cliente": ["SP", "RJ", "SP"],
    })  # fmt: skip
    f = Filtros(nota_max=2, data_inicio="2017-11-01", data_fim="2017-11-30", categorias=["cama mesa banho"])
    assert mascara(docs, f).tolist() == [True, False, True]
    assert Filtros().vazio() and Filtros().descrever() == "sem filtros"
