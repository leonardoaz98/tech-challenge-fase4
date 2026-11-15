"""Verificação das citações da resposta. Não depende de modelos nem da base."""

from src.generation.resposta import verificar_citacoes


def test_citacoes_fora_do_contexto_sao_removidas():
    texto, validas, invalidas = verificar_citacoes("Atrasos [R1][R2] e defeitos [R9].", n=3)
    assert validas == 2 and invalidas == ["[R9]"] and "[R9]" not in texto


def test_percentual_escrito_pelo_modelo_e_removido():
    from src.generation.resposta import verificar_numeros

    texto, removidos = verificar_numeros("Juntos somam 64.2% das reclamações.", set())
    assert removidos == ["64.2%"] and "64.2%" not in texto


def test_ranking_usa_os_numeros_da_contagem():
    from src.generation.resposta import ranking_temas

    freq = {"universo": 10543, "temas": [{"rotulo": "Produto não entregue", "avaliacoes": 3437, "percentual": 32.6}]}
    assert "3437 avaliações (32,6%)" in ranking_temas(freq) and "10.543" in ranking_temas(freq)
