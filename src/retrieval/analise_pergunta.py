"""Processamento da pergunta: escopo, tipo, filtros e reescrita.

Com o LLM, a pergunta do gestor ("o que reclamam da entrega?") vira:
- uma decisão de escopo (perguntas sobre outra coisa não chegam na busca);
- o tipo (frequência, exploração, evidência), que muda a estratégia de recuperação;
- filtros de metadados ("nota baixa" -> nota 1 a 2);
- 2 a 3 reescritas no vocabulário de quem escreve avaliação ("não chegou",
  "atrasou"), porque a pergunta do gestor e o texto do cliente usam palavras
  diferentes para a mesma coisa.

Sem chave da API, uma versão por regras faz o mínimo: tipo por palavras-chave,
a pergunta original mais o seu núcleo como consultas, e filtros só se vierem da
interface. Sem o LLM não há decisão de escopo: a recusa depende só do limiar.
"""

import re
from dataclasses import dataclass, field

from src.generation.llm import gerar_json
from src.retrieval.filtros import Filtros
from src.retrieval.nucleo import nucleo

STATUS = ["no_prazo", "atrasado", "nao_entregue"]


@dataclass
class Analise:
    no_escopo: bool
    tipo: str  # frequencia | exploratoria | evidencia
    tipo_tema: str  # problema | elogio | todos
    consultas: list[str]
    filtros: Filtros
    motivo: str = ""
    origem: str = "llm"
    uso: dict = field(default_factory=dict)


def _esquema(categorias: list[str]) -> dict:
    # Só construções básicas de JSON Schema; faixas e formato de data são validados depois.
    nulo_int = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    nulo_data = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["no_escopo", "motivo", "tipo", "tipo_tema", "consultas", "filtros"],
        "properties": {
            "no_escopo": {"type": "boolean"},
            "motivo": {"type": "string"},
            "tipo": {"type": "string", "enum": ["frequencia", "exploratoria", "evidencia"]},
            "tipo_tema": {"type": "string", "enum": ["problema", "elogio", "todos"]},
            "consultas": {"type": "array", "items": {"type": "string"}},
            "filtros": {
                "type": "object",
                "additionalProperties": False,
                "required": ["nota_min", "nota_max", "data_inicio", "data_fim", "categorias", "status_entrega"],
                "properties": {
                    "nota_min": nulo_int,
                    "nota_max": nulo_int,
                    "data_inicio": nulo_data,
                    "data_fim": nulo_data,
                    "categorias": {"type": "array", "items": {"type": "string", "enum": categorias}},
                    "status_entrega": {"type": "array", "items": {"type": "string", "enum": STATUS}},
                },
            },
        },
    }


SYSTEM = """Você prepara perguntas para um sistema de busca sobre avaliações de clientes de um e-commerce brasileiro (base pública Olist).

A base de conhecimento contém apenas avaliações escritas por clientes (título e comentário, em português informal), com metadados: nota de 1 a 5, data da compra (out/2016 a ago/2018), categoria do produto, status da entrega (no_prazo, atrasado, nao_entregue) e UF do cliente. Não há dados de faturamento, estoque, concorrentes, funcionários, nem nada fora da voz do cliente.

Para a pergunta recebida, decida:

1. no_escopo: true se a pergunta pode ser respondida a partir do que clientes escreveram nas avaliações. false se pede algo que essas avaliações não contêm (ex.: faturamento, previsão do tempo, dados de outra empresa, conhecimento geral). Em "motivo", explique em uma frase.

2. tipo:
   - "frequencia": pede ranking, "principais", "mais comuns", "com maior frequência", padrões recorrentes.
   - "evidencia": pede quais avaliações sustentam uma conclusão.
   - "exploratoria": o resto (o que relatam sobre X, como descrevem Y).

3. tipo_tema: "problema" se o foco é reclamação/insatisfação, "elogio" se é o que valorizam/elogiam, "todos" se ambos ou neutro.

4. consultas: 2 ou 3 frases curtas no vocabulário de um cliente escrevendo a avaliação, cobrindo formas diferentes de dizer a mesma coisa. Ex.: para "problemas de entrega": "produto chegou atrasado, demorou muito", "não recebi o produto até hoje". Não repita a pergunta original.

5. filtros: só o que a pergunta pede explicitamente ou implica sem ambiguidade.
   - "nota baixa", "insatisfeitos", "detratores": nota_max 2. "nota alta", "satisfeitos": nota_min 4.
   - Não use filtro de nota só porque a pergunta fala de reclamação: clientes nota 3 a 5 também reclamam.
   - Período: datas AAAA-MM-DD da compra. Categorias: use apenas nomes da lista permitida.
   - "entrega insatisfatória/atrasada": status_entrega ["atrasado", "nao_entregue"].
   - Sem filtro: null ou lista vazia."""


def analisar_com_llm(pergunta: str, categorias: list[str]) -> Analise:
    dados, uso = gerar_json(SYSTEM, f"Pergunta: {pergunta}", _esquema(categorias), esforco="low", max_tokens=4000)
    f = dados["filtros"]
    for chave in ("nota_min", "nota_max"):
        if f.get(chave) is not None:
            f[chave] = min(5, max(1, int(f[chave])))
    for chave in ("data_inicio", "data_fim"):
        if f.get(chave) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", f[chave]):
            f[chave] = None
    return Analise(
        no_escopo=dados["no_escopo"],
        tipo=dados["tipo"],
        tipo_tema=dados["tipo_tema"],
        consultas=[pergunta] + [c for c in dados["consultas"] if c.strip()][:3],
        filtros=Filtros.de_dict(f),
        motivo=dados["motivo"],
        uso=uso,
    )


_FREQUENCIA = re.compile(
    r"principa|mais comu|mais frequ|maior frequ|recorrent|padr(ao|ões|oes)|ranking|mais (elogi|reclam|relat)"
    r"|^\s*quais (s[aã]o |foram )?(os |as )?(problemas|reclama\w*|elogios|temas|aspectos)\b",
    re.I,
)
_EVIDENCIA = re.compile(r"quais avalia\w+ sustenta|evid[eê]ncia", re.I)
_ELOGIO = re.compile(r"elogi|valoriz|positiv|satisfeit|gost", re.I)
_PROBLEMA = re.compile(r"problem|reclam|insatisf|ruim|negativ|defeit|atras", re.I)


def analisar_por_regras(pergunta: str, filtros: Filtros | None = None) -> Analise:
    tipo = "evidencia" if _EVIDENCIA.search(pergunta) else "frequencia" if _FREQUENCIA.search(pergunta) else "exploratoria"
    tipo_tema = "elogio" if _ELOGIO.search(pergunta) and not _PROBLEMA.search(pergunta) else "problema" if _PROBLEMA.search(pergunta) else "todos"
    consultas = [pergunta] + ([nucleo(pergunta)] if nucleo(pergunta) != pergunta.strip().rstrip("?. ") else [])
    return Analise(True, tipo, tipo_tema, consultas, filtros or Filtros(), "Análise por regras (sem LLM).", origem="regras")
