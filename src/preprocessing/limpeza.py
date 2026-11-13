"""Funções de limpeza de texto.

Duas versões de cada texto:
- texto: limpo para leitura e para o modelo de embeddings (mantém acentos, caixa
  e pontuação, que carregam sentido em comentários informais: "NÃO RECEBI!!!").
- texto_busca: normalizado para a busca por palavra-chave (minúsculas, sem
  acento, sem pontuação), para "nao recebi" encontrar "Não recebi".
"""

import re
import unicodedata

from unidecode import unidecode

# Stopwords curtas do português. Lista própria e pequena de propósito: palavras como
# "não", "nada" e "nunca" mudam o sentido de uma reclamação e precisam ficar.
STOPWORDS = set(
    """a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela
    pelos pelas para pra pro com e ou que se ao aos à às é foi era ser estar esta
    este isso isto esse essa me meu minha eu voce você ele ela eles elas lhe já
    mas mais muito muita bem tambem também so só ate até quando como""".split()
)
# A lista é comparada com o texto já sem acento.
STOPWORDS = {unidecode(p) for p in STOPWORDS} - {"nao", "nada", "nunca", "sem"}

_ESPACOS = re.compile(r"\s+")
_LETRAS = re.compile(r"[a-zA-ZÀ-ÿ]+")
_NAO_ALFANUM = re.compile(r"[^a-z0-9 ]+")


def limpar(texto: str | None) -> str:
    """Normaliza unicode, troca quebras de linha por espaço e tira espaços sobrando."""
    if texto is None or (isinstance(texto, float)):
        return ""
    texto = unicodedata.normalize("NFC", str(texto))
    texto = texto.replace("\r", " ").replace("\n", " ")
    return _ESPACOS.sub(" ", texto).strip()


def juntar_titulo_e_comentario(titulo: str, comentario: str) -> str:
    """Título e comentário viram um texto só. O título costuma ser um resumo ("Péssimo")."""
    titulo, comentario = limpar(titulo), limpar(comentario)
    if titulo and comentario:
        separador = "" if titulo[-1] in ".!?" else "."
        return f"{titulo}{separador} {comentario}"
    return titulo or comentario


def contar_palavras(texto: str) -> int:
    """Conta palavras com letras (emojis e pontuação não contam)."""
    return len(_LETRAS.findall(texto))


def normalizar_para_busca(texto: str) -> str:
    texto = unidecode(texto.lower())
    return _ESPACOS.sub(" ", _NAO_ALFANUM.sub(" ", texto)).strip()


def tokenizar(texto: str) -> list[str]:
    """Tokens para o BM25: normalizados, sem stopwords, com pelo menos 2 caracteres."""
    return [t for t in normalizar_para_busca(texto).split() if len(t) > 1 and t not in STOPWORDS]
