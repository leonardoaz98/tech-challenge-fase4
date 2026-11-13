"""Classificação de temas por regras. Não depende de modelos nem da base."""

from src.preprocessing.limpeza import normalizar_para_busca as norm
from src.temas.temas import classificar_texto


def test_temas_basicos_e_negacao():
    assert "atraso" in classificar_texto(norm("O produto chegou muito atrasado"))
    assert "nao_entregue" in classificar_texto(norm("Até hoje não recebi meu pedido"))
    assert "incompleto" in classificar_texto(norm("Comprei 2 e foi entregue apenas 1"))
    assert "entrega_rapida" in classificar_texto(norm("Entrega rápida, chegou antes do prazo"))
    # "não chegou no prazo" é reclamação, não elogio
    assert "entrega_rapida" not in classificar_texto(norm("O produto não chegou no prazo"))
