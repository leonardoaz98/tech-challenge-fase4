"""Logger padrão do projeto: console + arquivo em logs/."""

import logging
import sys

from src import config


def obter_logger(nome: str) -> logging.Logger:
    log = logging.getLogger(nome)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    formato = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formato)
    log.addHandler(console)

    config.LOGS.mkdir(parents=True, exist_ok=True)
    arquivo = logging.FileHandler(config.LOGS / "pipeline.log", encoding="utf-8")
    arquivo.setFormatter(formato)
    log.addHandler(arquivo)
    log.propagate = False
    return log
