"""Filtros extraídos do texto da pergunta. Não depende do índice."""

from src.retrieval.filtros import extrair_filtros

CATEGORIAS = ["beleza saude", "cama mesa banho", "moveis decoracao", "moveis escritorio", "relogios presentes"]


def test_nota_baixa_e_explicita():
    assert extrair_filtros("Problemas nas avaliações com nota baixa?", CATEGORIAS)[0].nota_max == 2
    f, _ = extrair_filtros("Quem teve entrega atrasada mas deu nota 5?", CATEGORIAS)
    assert (f.nota_min, f.nota_max, f.status_entrega) == (5, 5, ["atrasado"])


def test_insatisfeito_nao_vira_satisfeito():
    f, _ = extrair_filtros("O que dizem os clientes insatisfeitos?", CATEGORIAS)
    assert f.nota_max == 2 and f.nota_min is None


def test_mes_e_ano():
    f, _ = extrair_filtros("Reclamações sobre compras feitas em novembro de 2017", CATEGORIAS)
    assert (f.data_inicio, f.data_fim) == ("2017-11-01", "2017-11-30")


def test_categoria_inteira_tem_prioridade_sobre_termo_principal():
    f, _ = extrair_filtros("Problemas em móveis e decoração", CATEGORIAS)
    assert f.categorias == ["moveis decoracao"]
    f, _ = extrair_filtros("O que dizem quem comprou relógios?", CATEGORIAS)
    assert f.categorias == ["relogios presentes"]


def test_sem_ancora_sem_filtro():
    assert extrair_filtros("O que os clientes falam sobre a nota fiscal?", CATEGORIAS)[0].vazio()
