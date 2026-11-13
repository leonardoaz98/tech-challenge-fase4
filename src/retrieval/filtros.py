"""Filtros de metadados (desafio adicional).

Os filtros são aplicados ANTES da busca (pré-filtragem): a busca roda só dentro do
subconjunto pedido. Filtrar depois (pós-filtragem) faria uma pergunta sobre uma
categoria pequena voltar quase vazia, porque os 100 mais parecidos seriam de
outras categorias.
"""

from dataclasses import asdict, dataclass, field

import pandas as pd


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
