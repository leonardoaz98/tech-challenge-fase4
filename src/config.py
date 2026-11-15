"""Configuração central: caminhos, modelos e parâmetros do pipeline.

Tudo que muda o resultado fica aqui, num lugar só, para a execução ser reproduzível
e para cada parâmetro ter uma justificativa rastreável no README.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")  # OLLAMA_HOST / OLLAMA_MODEL opcionais

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
LIMIAR_RELEVANCIA = -1.75  # calibrado em reports/calibracao_limiar.json; abaixo disso a avaliação não conta como evidência
MIN_EVIDENCIAS = 3  # menos que isso = evidência insuficiente
MIN_UNIVERSO_FREQUENCIA = 30  # abaixo disso, cada avaliação mexe 3+ pontos percentuais na contagem de temas

# --- LLM local (Ollama) ---------------------------------------------------
# Variáveis de ambiente (ou .env) permitem trocar o modelo ou o endereço sem mudar código.
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODELO_LLM = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
CONTEXTO_LLM = 8192  # tokens; 15 evidências + tabela de temas + instruções cabem com folga
TIMEOUT_LLM = 600  # segundos; em CPU sem GPU uma resposta pode levar minutos
# Em GPUs com pouca VRAM livre (ex.: 6 GB compartilhados no WSL), limitar as camadas
# na GPU evita o "cudaMalloc out of memory". Sem a variável, o Ollama decide sozinho.
_num_gpu = os.getenv("OLLAMA_NUM_GPU")
NUM_GPU_LLM = int(_num_gpu) if _num_gpu else None
_num_batch = os.getenv("OLLAMA_NUM_BATCH")
NUM_BATCH_LLM = int(_num_batch) if _num_batch else None
