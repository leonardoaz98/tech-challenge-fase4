"""Chamada ao LLM local (Ollama) com saída em JSON validado por esquema.

- Modelo: config.MODELO_LLM (padrão qwen2.5:7b), servido pelo Ollama em config.OLLAMA_URL.
- Saída estruturada: o esquema JSON vai no campo `format` e o Ollama restringe a
  geração a ele, então a resposta já vem como JSON no formato pedido.
- Temperatura 0: a mesma pergunta gera a mesma resposta, o que torna a avaliação
  reproduzível.
- Sem custo por chamada e sem chave: roda na máquina de quem executa.
"""

import json
import time
import urllib.error
import urllib.request

from src import config


class LLMIndisponivel(RuntimeError):
    """Ollama fora do ar ou modelo não baixado: o pipeline roda só a recuperação."""


def _get(caminho: str, timeout: float) -> dict:
    with urllib.request.urlopen(config.OLLAMA_URL + caminho, timeout=timeout) as r:
        return json.loads(r.read())


def disponivel() -> bool:
    """Ollama respondendo e com o modelo configurado já baixado."""
    try:
        modelos = {m["name"] for m in _get("/api/tags", timeout=2).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return config.MODELO_LLM in modelos or f"{config.MODELO_LLM}:latest" in modelos


def gerar_json(system: str, mensagem: str, esquema: dict, max_tokens: int = 2000) -> tuple[dict, dict]:
    """Retorna (json da resposta, metadados de uso)."""
    opcoes = {"temperature": 0, "seed": 42, "num_ctx": config.CONTEXTO_LLM, "num_predict": max_tokens}
    if config.NUM_GPU_LLM is not None:  # limita camadas na GPU em máquinas com pouca VRAM livre
        opcoes["num_gpu"] = config.NUM_GPU_LLM
    if config.NUM_BATCH_LLM is not None:
        opcoes["num_batch"] = config.NUM_BATCH_LLM
    corpo = {
        "model": config.MODELO_LLM,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": mensagem}],
        "format": esquema,
        "stream": False,
        "options": opcoes,
        "keep_alive": "30m",
    }
    pedido = urllib.request.Request(
        config.OLLAMA_URL + "/api/chat", data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(pedido, timeout=config.TIMEOUT_LLM) as r:
            resposta = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise LLMIndisponivel(f"Ollama recusou a chamada ({e.code}): {e.read().decode()[:300]}") from e
    except (urllib.error.URLError, OSError) as e:
        raise LLMIndisponivel(f"Ollama indisponível em {config.OLLAMA_URL}: {e}") from e

    if resposta.get("done_reason") == "length":
        raise RuntimeError("Resposta cortada pelo limite de tokens (max_tokens).")
    uso = {
        "modelo": resposta.get("model"),
        "entrada": resposta.get("prompt_eval_count"),
        "saida": resposta.get("eval_count"),
        "segundos": round(time.time() - t0, 1),
    }
    return json.loads(resposta["message"]["content"]), uso
