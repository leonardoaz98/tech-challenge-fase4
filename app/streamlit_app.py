"""Interface do Voice of Customer Intelligence.

Uso: streamlit run app/streamlit_app.py
"""

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation import llm  # noqa: E402
from src.pipeline import responder  # noqa: E402
from src.retrieval.filtros import Filtros  # noqa: E402
from src.retrieval.indice import carregar  # noqa: E402

st.set_page_config(page_title="Voz do Cliente · Olist", page_icon="💬", layout="wide")

EXEMPLOS = [
    "Quais são os principais problemas relatados pelos clientes?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    "O que os clientes mais elogiam na categoria cama mesa banho?",
    "Qual foi o faturamento da Olist em 2017?",
]
STATUS = {
    "respondida": ("success", "Resposta fundamentada nas avaliações"),
    "so_recuperacao": ("info", "Modo só recuperação"),
    "fora_do_escopo": ("warning", "Fora do escopo da base"),
    "evidencia_insuficiente": ("warning", "Evidência insuficiente"),
}


@st.cache_resource(show_spinner="Carregando índice e modelos...")
def indice():
    return carregar()


idx = indice()

with st.sidebar:
    st.header("Filtros (opcional)")
    st.caption("Se nada for marcado, o sistema extrai os filtros da própria pergunta.")
    notas = st.slider("Nota da avaliação", 1, 5, (1, 5))
    periodo = st.date_input("Período da compra", value=(), min_value=pd.Timestamp("2016-09-01"), max_value=pd.Timestamp("2018-10-31"))
    categorias = st.multiselect("Categoria do produto", idx.categorias)
    entrega = st.multiselect("Entrega", ["no_prazo", "atrasado", "nao_entregue"])
    st.divider()
    if llm.disponivel():
        st.success("LLM: Claude (API Anthropic)")
    else:
        st.warning("Sem ANTHROPIC_API_KEY: só recuperação, sem resposta gerada.")
    st.caption(f"{len(idx.docs):,} avaliações na base de conhecimento".replace(",", "."))

st.title("💬 Voz do Cliente")
st.caption("Pergunte em linguagem natural sobre as avaliações de clientes da Olist (2016–2018). Cada afirmação aponta as avaliações que a sustentam.")

exemplo = st.selectbox("Exemplos", [""] + EXEMPLOS, format_func=lambda x: x or "Escolha um exemplo ou escreva abaixo")
pergunta = st.text_input("Pergunta", value=exemplo)

if st.button("Perguntar", type="primary", disabled=not pergunta.strip()):
    filtros = Filtros(
        nota_min=notas[0] if notas != (1, 5) else None,
        nota_max=notas[1] if notas != (1, 5) else None,
        data_inicio=str(periodo[0]) if len(periodo) == 2 else None,
        data_fim=str(periodo[1]) if len(periodo) == 2 else None,
        categorias=categorias,
        status_entrega=entrega,
    )
    with st.spinner("Buscando avaliações e gerando a resposta..."):
        r = responder(pergunta, filtros)

    tipo, titulo = STATUS[r.status]
    getattr(st, tipo)(f"**{titulo}**")
    st.markdown(r.mensagem)

    if r.resposta and r.resposta.lacunas:
        st.caption(f"Limitações apontadas pelo modelo: {r.resposta.lacunas}")

    if r.frequencias:
        st.subheader("Frequência de temas na base")
        st.caption(f"Contagem sobre {r.frequencias['universo']:,} avaliações (base inteira com os filtros), não só sobre as recuperadas.".replace(",", "."))
        freq = pd.DataFrame(r.frequencias["temas"]).query("avaliacoes > 0")
        st.bar_chart(freq.set_index("rotulo")["percentual"], horizontal=True)

    if r.resposta and r.resposta.insights:
        st.subheader("Insights e evidências")
        evid = r.busca.evidencias
        for ins in r.resposta.insights:
            with st.expander(ins["insight"]):
                for ref in ins["evidencias"]:
                    e = evid[int(ref[1:]) - 1]
                    st.markdown(f"**[{ref}]** ⭐ {e.nota} · {e.categoria} · {e.status_entrega} · compra {e.data_compra}\n\n> {e.texto}")

    if r.busca and r.busca.evidencias:
        st.subheader("Avaliações recuperadas")
        tabela = pd.DataFrame([{**e.como_dict(), "ref": f"R{i}"} for i, e in enumerate(r.busca.evidencias, 1)])
        st.dataframe(tabela[["ref", "texto", "nota", "categoria", "status_entrega", "data_compra", "score_rerank", "review_id"]],
                     hide_index=True, width="stretch")  # fmt: skip

    with st.expander("Por dentro do pipeline"):
        a = r.analise
        st.markdown(f"**1. Processamento da pergunta** ({a.origem}) · escopo: `{a.no_escopo}` · tipo: `{a.tipo}` · tema: `{a.tipo_tema}`")
        st.markdown(f"Filtros: `{a.filtros.descrever()}`")
        if len(a.consultas) > 1:
            st.markdown("Reescritas usadas na busca:\n" + "\n".join(f"- {c}" for c in a.consultas[1:]))
        if r.busca:
            b = r.busca
            st.markdown(
                f"**2. Recuperação** · universo após filtros: {b.universo:,} · candidatos densos: {b.candidatos_denso} · "
                f"BM25: {b.candidatos_bm25} · após fusão RRF: {b.candidatos_fundidos} · descartados pelo limiar: {b.descartadas_limiar} · "
                f"maior score do reranker: {b.max_score}"
            )
        if r.resposta:
            st.markdown(f"**3. Geração** · citações válidas: {r.resposta.citacoes_validas} · inválidas removidas: {len(r.resposta.citacoes_invalidas)}")
            st.text_area("Contexto enviado ao modelo", r.resposta.contexto, height=300)
        st.json({"tempos_s": r.tempos, "uso_llm": {"analise": a.uso, "resposta": r.resposta.uso if r.resposta else {}}})
