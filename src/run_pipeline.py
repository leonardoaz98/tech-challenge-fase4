"""Reconstrói tudo do zero, na ordem: dados -> base -> temas -> índice -> calibração -> avaliação.

As etapas sem LLM rodam sempre. As que usam o Claude só rodam com ANTHROPIC_API_KEY.

Uso:
  python -m src.run_pipeline           # tudo que der com o que estiver configurado
  python -m src.run_pipeline --juiz    # inclui avaliação de recuperação e fidelidade (LLM juiz)
"""

import argparse
import json
import time
from datetime import datetime

from src import config
from src.evaluation import avaliar, calibrar_limiar
from src.generation import llm
from src.indexing import indexar
from src.ingestion import baixar_dados
from src.logger import obter_logger
from src.preprocessing import base_conhecimento
from src.temas import temas

log = obter_logger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--juiz", action="store_true")
    args = ap.parse_args()

    etapas = [
        ("baixar_dados", baixar_dados.baixar),
        ("base_conhecimento", base_conhecimento.construir),
        ("temas", temas.classificar_base),
        ("indice", indexar.indexar),
        ("calibracao_limiar", calibrar_limiar.calibrar),
    ]
    registro = {"inicio": datetime.now().isoformat(timespec="seconds"), "llm": llm.disponivel(), "etapas": {}}
    for nome, funcao in etapas:
        t = time.time()
        log.info(">> %s", nome)
        funcao()
        registro["etapas"][nome] = round(time.time() - t, 1)

    t = time.time()
    import sys

    sys.argv = ["avaliar"] + (["--juiz"] if args.juiz else [])
    avaliar.main()
    registro["etapas"]["avaliacao"] = round(time.time() - t, 1)

    with open(config.RELATORIOS / "execucoes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    log.info("Concluído: %s", registro["etapas"])


if __name__ == "__main__":
    main()
