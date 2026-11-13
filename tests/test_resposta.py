"""Verificação das citações da resposta. Não depende de modelos nem da base."""

from src.generation.resposta import verificar_citacoes


def test_citacoes_fora_do_contexto_sao_removidas():
    texto, validas, invalidas = verificar_citacoes("Atrasos [R1][R2] e defeitos [R9].", n=3)
    assert validas == 2 and invalidas == ["[R9]"] and "[R9]" not in texto
