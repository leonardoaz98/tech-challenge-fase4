"""Baixa a base pública da Olist do Kaggle e copia os CSVs usados para data/raw/.

A base é pública: o kagglehub baixa sem conta nem chave. Se o download falhar
(rede, proxy), baixe o zip manualmente em
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce e extraia em data/raw/.

Uso: python -m src.ingestion.baixar_dados
"""

import shutil
from pathlib import Path

from src import config
from src.logger import obter_logger

log = obter_logger(__name__)


def baixar() -> Path:
    config.DADOS_BRUTOS.mkdir(parents=True, exist_ok=True)
    faltando = [a for a in config.ARQUIVOS_OLIST if not (config.DADOS_BRUTOS / a).exists()]
    if not faltando:
        log.info("Base já está em %s", config.DADOS_BRUTOS)
        return config.DADOS_BRUTOS

    import kagglehub

    origem = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    for arquivo in faltando:
        shutil.copy2(origem / arquivo, config.DADOS_BRUTOS / arquivo)
        log.info("Copiado %s", arquivo)
    return config.DADOS_BRUTOS


if __name__ == "__main__":
    baixar()
