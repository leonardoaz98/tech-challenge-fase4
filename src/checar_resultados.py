"""Confere se a base de conhecimento e os temas regerados batem com a referência versionada.

Uma mudança no código que altere o número de documentos ou a contagem de temas
quebra o CI. Se a mudança foi de propósito, grave a referência nova num commit
próprio, explicando o motivo:  python -m src.checar_resultados --gravar
"""

import json
import sys

from src import config

REFERENCIA = config.RELATORIOS / "referencia.json"


def atual() -> dict:
    funil = json.loads(config.FUNIL.read_text(encoding="utf-8"))
    temas = json.loads((config.RELATORIOS / "temas_resumo.json").read_text(encoding="utf-8"))
    return {
        "documentos": funil["base_de_conhecimento"]["documentos"],
        "removidas": {k: v["removidas"] for k, v in funil.items() if isinstance(v, dict) and "removidas" in v},
        "por_nota": funil["base_de_conhecimento"]["por_nota"],
        "temas": temas["por_tema"],
    }


def main() -> None:
    agora = atual()
    if "--gravar" in sys.argv:
        REFERENCIA.write_text(json.dumps(agora, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Referência gravada.")
        return
    ref = json.loads(REFERENCIA.read_text(encoding="utf-8"))
    if json.dumps(agora, sort_keys=True) != json.dumps(ref, sort_keys=True):
        print("Resultado diferente da referência:")
        print(json.dumps({"referencia": ref, "atual": agora}, ensure_ascii=False, indent=1))
        sys.exit(1)
    print("Funil e temas batem com a referência.")


if __name__ == "__main__":
    main()
