"""Chamada ao Claude (API da Anthropic) com saída em JSON validado por esquema.

- Modelo: config.MODELO_LLM.
- Saída estruturada (`output_config.format`): a resposta já vem como JSON válido no
  esquema pedido, sem parsing frágil de texto.
- `fallbacks: "default"`: se o classificador de segurança recusar a pergunta, a API
  tenta de novo num modelo alternativo na mesma chamada.
- Cache de prompt no system: as instruções são iguais em toda pergunta.
"""

import json
import os
from functools import lru_cache

from dotenv import load_dotenv

from src import config

load_dotenv(config.RAIZ / ".env")


class LLMIndisponivel(RuntimeError):
    """Sem credencial da Anthropic: o pipeline roda só a recuperação."""


def disponivel() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


@lru_cache(maxsize=1)
def _cliente():
    import anthropic

    if not disponivel():
        raise LLMIndisponivel("Defina ANTHROPIC_API_KEY no arquivo .env (ver .env.example).")
    return anthropic.Anthropic()


def gerar_json(system: str, mensagem: str, esquema: dict, esforco: str, max_tokens: int = 8000) -> tuple[dict, dict]:
    """Retorna (json da resposta, metadados de uso)."""
    import anthropic

    try:
        resposta = _cliente().beta.messages.create(
            model=config.MODELO_LLM,
            max_tokens=max_tokens,
            betas=[config.BETA_FALLBACK],
            fallbacks="default",
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": mensagem}],
            output_config={"effort": esforco, "format": {"type": "json_schema", "schema": esquema}},
        )
    except anthropic.AuthenticationError as e:
        raise LLMIndisponivel("Chave da Anthropic inválida.") from e

    if resposta.stop_reason == "refusal":
        raise RuntimeError(f"O modelo recusou a solicitação: {resposta.stop_details}")
    if resposta.stop_reason == "max_tokens":
        raise RuntimeError("Resposta cortada por max_tokens.")
    texto = next(b.text for b in resposta.content if b.type == "text")
    uso = {
        "modelo": resposta.model,
        "entrada": resposta.usage.input_tokens,
        "saida": resposta.usage.output_tokens,
        "cache_lido": resposta.usage.cache_read_input_tokens or 0,
        "request_id": resposta._request_id,
    }
    return json.loads(texto), uso
