"""Limpeza e tokenização do texto. Não depende de modelos nem da base."""

from src.preprocessing.limpeza import contar_palavras, juntar_titulo_e_comentario, limpar, tokenizar


def test_limpeza_quebra_de_linha_e_espacos():
    assert limpar("Produto  bom\r\nchegou   rápido ") == "Produto bom chegou rápido"
    assert limpar(None) == "" and limpar(float("nan")) == ""


def test_titulo_e_comentario():
    assert juntar_titulo_e_comentario("Péssimo", "não chegou") == "Péssimo. não chegou"
    assert juntar_titulo_e_comentario("Ótimo!", "recomendo") == "Ótimo! recomendo"
    assert juntar_titulo_e_comentario(None, "só comentário") == "só comentário"


def test_contagem_ignora_emoji_e_pontuacao():
    assert contar_palavras("Obrigada 👏🏻👍🏻 !!!") == 1


def test_tokenizacao_mantem_negacao():
    tokens = tokenizar("Não recebi o produto")
    assert "nao" in tokens and "recebi" in tokens and "o" not in tokens
