"""Filtros de metadados (desafio adicional).

Os filtros são aplicados ANTES da busca (pré-filtragem): a busca roda só dentro do
subconjunto pedido. Filtrar depois (pós-filtragem) faria uma pergunta sobre uma
categoria pequena voltar quase vazia, porque os 100 mais parecidos seriam de
outras categorias.
"""

import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from src.preprocessing.limpeza import STOPWORDS, normalizar_para_busca


@dataclass
class Filtros:
    nota_min: int | None = None
    nota_max: int | None = None
    data_inicio: str | None = None  # AAAA-MM-DD, data da compra
    data_fim: str | None = None
    categorias: list[str] = field(default_factory=list)
    status_entrega: list[str] = field(default_factory=list)  # no_prazo | atrasado | nao_entregue
    ufs: list[str] = field(default_factory=list)

    def vazio(self) -> bool:
        return not any(v for v in asdict(self).values())

    def descrever(self) -> str:
        partes = []
        if self.nota_min is not None or self.nota_max is not None:
            partes.append(f"nota {self.nota_min or 1} a {self.nota_max or 5}")
        if self.data_inicio or self.data_fim:
            partes.append(f"compras de {self.data_inicio or 'início'} a {self.data_fim or 'fim'}")
        if self.categorias:
            partes.append("categoria: " + ", ".join(self.categorias))
        if self.status_entrega:
            partes.append("entrega: " + ", ".join(self.status_entrega))
        if self.ufs:
            partes.append("UF: " + ", ".join(self.ufs))
        return "; ".join(partes) or "sem filtros"

    @classmethod
    def de_dict(cls, d: dict | None) -> "Filtros":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and v not in (None, "")})


def mascara(docs: pd.DataFrame, f: Filtros) -> pd.Series:
    m = pd.Series(True, index=docs.index)
    if f.nota_min is not None:
        m &= docs["nota"] >= f.nota_min
    if f.nota_max is not None:
        m &= docs["nota"] <= f.nota_max
    if f.data_inicio:
        m &= docs["data_compra"] >= pd.Timestamp(f.data_inicio)
    if f.data_fim:
        m &= docs["data_compra"] <= pd.Timestamp(f.data_fim) + pd.Timedelta(days=1)
    if f.categorias:
        alvo = {c.lower() for c in f.categorias}
        m &= docs["categorias"].apply(lambda cs: any(c in alvo for c in cs))
    if f.status_entrega:
        m &= docs["status_entrega"].isin(f.status_entrega)
    if f.ufs:
        m &= docs["uf_cliente"].isin([u.upper() for u in f.ufs])
    return m


# --- Filtros a partir do texto da pergunta ---------------------------------
# Regras determinísticas, não o LLM: com um modelo local de 7B, a extração de
# filtros pelo modelo inventava recortes que a pergunta não pedia (período da base
# inteira, status de entrega). Regras são previsíveis, testáveis (tests/test_filtros.py)
# e funcionam também sem LLM. Cada filtro precisa de uma âncora no texto.

_MESES = ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho",
          "agosto", "setembro", "outubro", "novembro", "dezembro"]  # fmt: skip
_NOTA_EXPLICITA = re.compile(r"\bnotas? ([1-5])(?: (?:a|e|ou) ([1-5]))?\b")
_NOTA_BAIXA = re.compile(r"\bnotas? baixas?\b|\binsatisfeit|\bdetrator|\bavaliac\w+ negativas?")
_NOTA_ALTA = re.compile(r"\bnotas? altas?\b|\bsatisfeit|\bpromotor|\bavaliac\w+ positivas?")
_MES_ANO = re.compile(r"\b(" + "|".join(_MESES) + r") de (201[6-8])\b")
_ANO = re.compile(r"\b(?:em|de|durante) (201[6-8])\b")
_INSATISFATORIA = re.compile(r"\bentregas? insatisfat")
_ATRASADA = re.compile(r"\batrasad")
_NAO_ENTREGUE = re.compile(r"\bnao (?:foi |foram )?(?:entregue|chegou|chegaram|recebe)")
_NO_PRAZO = re.compile(r"\b(?:no|dentro do|antes do) prazo\b")


def _categorias_citadas(tokens: set[str], categorias: list[str]) -> list[str]:
    """Categoria citada: todos os seus termos aparecem na pergunta ("móveis e decoração").
    Se nenhuma aparece inteira, vale o termo principal (o primeiro, com 5+ letras:
    "relógios" -> relogios presentes; "móveis" -> todas as de móveis)."""
    inteiras, por_termo = [], []
    for c in categorias:
        termos = [t for t in c.split() if t not in STOPWORDS and not t.isdigit()]
        if termos and set(termos) <= tokens:
            inteiras.append(c)
        elif termos and len(termos[0]) >= 5 and termos[0] in tokens:
            por_termo.append(c)
    return inteiras or por_termo


def extrair_filtros(pergunta: str, categorias: list[str]) -> tuple[Filtros, set[str]]:
    """Retorna os filtros ancorados no texto e os termos da pergunta que viraram filtro
    (para saber se sobra um assunto depois de tirar o recorte)."""
    texto = normalizar_para_busca(pergunta)
    f, usados = Filtros(), set()

    if m := _NOTA_EXPLICITA.search(texto):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        f.nota_min, f.nota_max = min(a, b), max(a, b)
        usados |= {"nota", "notas"}
    elif m := _NOTA_BAIXA.search(texto):
        f.nota_max = 2
        usados |= set(m.group(0).split()) | {"insatisfeitos", "insatisfeitas", "insatisfeito"}
    elif m := _NOTA_ALTA.search(texto):
        f.nota_min = 4
        usados |= set(m.group(0).split()) | {"satisfeitos", "satisfeitas", "satisfeito"}

    if m := _MES_ANO.search(texto):
        mes, ano = _MESES.index(m.group(1)) + 1, int(m.group(2))
        fim = pd.Timestamp(ano, mes, 1) + pd.offsets.MonthEnd(0)
        f.data_inicio, f.data_fim = f"{ano}-{mes:02d}-01", str(fim.date())
        usados |= {m.group(1), m.group(2), "compras", "feitas"}
    elif m := _ANO.search(texto):
        f.data_inicio, f.data_fim = f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
        usados |= {m.group(1)}

    if _INSATISFATORIA.search(texto):
        f.status_entrega = ["atrasado", "nao_entregue"]
        usados |= {"entrega", "entregas", "insatisfatoria", "insatisfatorias", "experiencias", "experiencia"}
    else:
        if _ATRASADA.search(texto):
            f.status_entrega.append("atrasado")
            usados |= {"entrega", "atrasada", "atrasadas", "atrasado", "atrasados"}
        if _NAO_ENTREGUE.search(texto):
            f.status_entrega.append("nao_entregue")
        if _NO_PRAZO.search(texto):
            f.status_entrega.append("no_prazo")

    tokens = set(texto.split())
    f.categorias = _categorias_citadas(tokens, categorias)
    if f.categorias:
        usados |= {t for c in f.categorias for t in c.split()} | {"categoria"}
    return f, usados
