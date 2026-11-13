"""Constrói a base de conhecimento: uma linha por avaliação com texto útil + metadados.

Etapas (cada descarte é contado em reports/funil.json):
1. Seleção: avaliações com título ou comentário.
2. Limpeza: unicode, quebras de linha, espaços; título + comentário num texto só.
3. Ausentes: sem texto = fora da base (a nota segue nas estatísticas gerais);
   metadados ausentes viram categoria explícita ("sem_categoria", "nao_entregue").
4. Organização: um documento por review_id (o mesmo review_id aparece em até 3
   pedidos com o mesmo texto; contar uma vez evita inflar frequências).
5. Informatividade: textos com menos de config.MIN_PALAVRAS palavras saem do índice.

Uso: python -m src.preprocessing.base_conhecimento
"""

import json

import numpy as np
import pandas as pd

from src import config
from src.logger import obter_logger
from src.preprocessing.limpeza import (
    contar_palavras,
    juntar_titulo_e_comentario,
    normalizar_para_busca,
)

log = obter_logger(__name__)


def _ler(nome: str) -> pd.DataFrame:
    return pd.read_csv(config.DADOS_BRUTOS / nome)


def _categoria_por_pedido() -> pd.DataFrame:
    """Categoria principal de cada pedido = categoria do item mais caro.

    727 pedidos têm mais de uma categoria. Ficar com a do item de maior valor
    representa o que o cliente mais provavelmente avaliou; a lista completa
    fica em `categorias` para o filtro não perder nenhum pedido.
    """
    itens = _ler("olist_order_items_dataset.csv")
    produtos = _ler("olist_products_dataset.csv")[["product_id", "product_category_name"]]
    itens = itens.merge(produtos, on="product_id", how="left")
    itens["product_category_name"] = itens["product_category_name"].fillna("sem_categoria")

    principal = (
        itens.sort_values("price", ascending=False)
        .drop_duplicates("order_id")[["order_id", "product_category_name"]]
        .rename(columns={"product_category_name": "categoria"})
    )
    todas = (
        itens.groupby("order_id")["product_category_name"]
        .agg(lambda s: sorted(set(s)))
        .rename("categorias")
        .reset_index()
    )
    resumo = itens.groupby("order_id").agg(
        n_itens=("order_item_id", "count"), valor_pedido=("price", "sum"), frete=("freight_value", "sum")
    )
    return principal.merge(todas, on="order_id").merge(resumo, on="order_id")


def _entrega_por_pedido() -> pd.DataFrame:
    pedidos = _ler("olist_orders_dataset.csv")
    clientes = _ler("olist_customers_dataset.csv")[["customer_id", "customer_state"]]
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        pedidos[col] = pd.to_datetime(pedidos[col])

    entregue = pedidos["order_delivered_customer_date"].notna()
    atraso = (
        pedidos["order_delivered_customer_date"].dt.normalize() - pedidos["order_estimated_delivery_date"]
    ).dt.days
    pedidos["atraso_dias"] = atraso  # NaN quando não houve entrega
    pedidos["status_entrega"] = np.select(
        [~entregue, atraso > 0], ["nao_entregue", "atrasado"], default="no_prazo"
    )
    pedidos["prazo_dias"] = (
        pedidos["order_delivered_customer_date"] - pedidos["order_purchase_timestamp"]
    ).dt.days
    pedidos = pedidos.merge(clientes, on="customer_id", how="left")
    return pedidos.rename(
        columns={
            "order_status": "status_pedido",
            "order_purchase_timestamp": "data_compra",
            "customer_state": "uf_cliente",
        }
    )[["order_id", "status_pedido", "data_compra", "atraso_dias", "prazo_dias", "status_entrega", "uf_cliente"]]


def construir() -> pd.DataFrame:
    funil: dict[str, dict] = {}

    reviews = _ler("olist_order_reviews_dataset.csv")
    funil["avaliacoes_na_base"] = {"linhas": len(reviews), "review_ids": reviews["review_id"].nunique()}

    # 1-2. Seleção e limpeza
    reviews["texto"] = [
        juntar_titulo_e_comentario(t, c)
        for t, c in zip(reviews["review_comment_title"], reviews["review_comment_message"])
    ]
    com_texto = reviews[reviews["texto"].str.len() > 0].copy()
    funil["sem_texto"] = {
        "removidas": len(reviews) - len(com_texto),
        "motivo": "Sem título nem comentário: só a nota. Continuam nas estatísticas de nota, fora da base de texto.",
    }

    # Metadados do pedido
    com_texto = com_texto.merge(_entrega_por_pedido(), on="order_id", how="left")
    com_texto = com_texto.merge(_categoria_por_pedido(), on="order_id", how="left")
    sem_itens = com_texto["categoria"].isna().sum()
    com_texto["categoria"] = com_texto["categoria"].fillna("sem_categoria")
    com_texto["categorias"] = com_texto["categorias"].apply(lambda c: c if isinstance(c, list) else ["sem_categoria"])

    # 4. Um documento por review_id
    antes = len(com_texto)
    com_texto = com_texto.sort_values("data_compra")
    pedidos_por_review = com_texto.groupby("review_id")["order_id"].nunique().rename("n_pedidos")
    docs = com_texto.drop_duplicates("review_id", keep="first").merge(pedidos_por_review, on="review_id")
    funil["review_id_repetido"] = {
        "removidas": antes - len(docs),
        "motivo": "O mesmo review_id aparece em mais de um pedido com o mesmo texto. Mantido 1 documento (pedido mais antigo); n_pedidos guarda quantos eram.",
    }

    # 5. Informatividade
    docs["n_palavras"] = docs["texto"].map(contar_palavras)
    pouco_texto = docs["n_palavras"] < config.MIN_PALAVRAS
    exemplos = docs.loc[pouco_texto, "texto"].str.lower().value_counts().head(10).to_dict()
    docs = docs[~pouco_texto].copy()
    funil["texto_curto_demais"] = {
        "removidas": int(pouco_texto.sum()),
        "motivo": f"Menos de {config.MIN_PALAVRAS} palavras: não explicam o porquê da nota.",
        "exemplos_mais_comuns": exemplos,
    }

    # Campos finais
    docs["data_avaliacao"] = pd.to_datetime(docs["review_creation_date"])
    docs["texto_busca"] = docs["texto"].map(normalizar_para_busca)
    docs["categoria"] = docs["categoria"].str.replace("_", " ")
    docs["categorias"] = docs["categorias"].apply(lambda cs: [c.replace("_", " ") for c in cs])
    docs = docs.rename(columns={"review_score": "nota"})
    colunas = [
        "review_id", "order_id", "texto", "texto_busca", "nota", "data_avaliacao", "data_compra",
        "status_pedido", "status_entrega", "atraso_dias", "prazo_dias", "uf_cliente",
        "categoria", "categorias", "n_itens", "valor_pedido", "frete", "n_pedidos", "n_palavras",
    ]  # fmt: skip
    docs = docs[colunas].sort_values("review_id").reset_index(drop=True)

    funil["base_de_conhecimento"] = {
        "documentos": len(docs),
        "pedidos_sem_itens_sem_categoria": int(sem_itens),
        "por_nota": docs["nota"].value_counts().sort_index().to_dict(),
        "por_status_entrega": docs["status_entrega"].value_counts().to_dict(),
        "periodo_avaliacoes": [str(docs["data_avaliacao"].min().date()), str(docs["data_avaliacao"].max().date())],
        "palavras_mediana": float(docs["n_palavras"].median()),
    }

    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    config.RELATORIOS.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(config.DOCUMENTOS, index=False)
    config.FUNIL.write_text(json.dumps(funil, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("Base de conhecimento: %d documentos -> %s", len(docs), config.DOCUMENTOS)
    return docs


if __name__ == "__main__":
    construir()
