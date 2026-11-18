"""Configuração central: caminhos, modelos e parâmetros do pipeline.

Tudo que muda o resultado fica aqui, num lugar só, para a execução ser reproduzível
e para cada parâmetro ter uma justificativa rastreável no README.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# --- Pastas ---------------------------------------------------------------
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
INDICE = RAIZ / "data" / "index"
RELATORIOS = RAIZ / "reports"
IMAGENS = RAIZ / "images"
LOGS = RAIZ / "logs"

DOCUMENTOS = DADOS_PROCESSADOS / "documentos.parquet"
FUNIL = RELATORIOS / "funil.json"
EMBEDDINGS = INDICE / "embeddings.npy"
INDICE_META = INDICE / "indice_meta.json"
TEMAS = DADOS_PROCESSADOS / "temas.parquet"

# --- Fonte ----------------------------------------------------------------
KAGGLE_DATASET = "olistbr/brazilian-ecommerce"
ARQUIVOS_OLIST = [
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_customers_dataset.csv",
    "product_category_name_translation.csv",
]

# --- Base de conhecimento -------------------------------------------------
# Texto com menos palavras que isso não explica nada ("ok", "bom", "recomendo").
# A nota continua contando nas estatísticas; só não vira evidência.
MIN_PALAVRAS = 3

# --- Modelos locais (rodam em CPU, sem chave) -----------------------------
MODELO_EMBEDDING = "intfloat/multilingual-e5-base"
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LOTE_EMBEDDING = 128

# --- Recuperação ----------------------------------------------------------
CANDIDATOS_DENSO = 100  # top-k da busca vetorial por consulta
CANDIDATOS_BM25 = 100  # top-k da busca por palavra-chave por consulta
RRF_K = 60  # constante da Reciprocal Rank Fusion (valor do artigo original)
CANDIDATOS_RERANK = 60  # quantos candidatos fundidos vão para o reranker
EVIDENCIAS_CONTEXTO = 15  # avaliações enviadas ao LLM
MMR_LAMBDA = 0.7  # 1 = só relevância, 0 = só diversidade
# Limiares sobre o score do reranker (logit). Calibrados em reports/calibracao_limiar.json.
LIMIAR_RELEVANCIA = 0.0  # abaixo disso a avaliação não conta como evidência
MIN_EVIDENCIAS = 3  # menos que isso = evidência insuficiente

# --- LLM (API da Anthropic) -----------------------------------------------
MODELO_LLM = "claude-opus-5-5"
ESFORCO_ANALISE = "low"  # entender a pergunta é tarefa simples
ESFORCO_RESPOSTA = "medium"  # síntese com citações
BETA_FALLBACK = "server-side-fallback-2026-07-01"
