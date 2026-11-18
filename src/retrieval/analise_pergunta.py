"""Processamento da pergunta: escopo, tipo, filtros e reescrita.

Com o LLM local (Ollama), a pergunta do gestor ("o que reclamam da entrega?") vira:
- o assunto (experiência de compra, dados da empresa, conhecimento geral, tarefa
  de texto): só "experiência de compra" chega na busca;
- o tipo (frequência, exploração, evidência), que muda a estratégia de recuperação;
- 2 a 3 reescritas no vocabulário de quem escreve avaliação ("não chegou",
  "atrasou"), porque a pergunta do gestor e o texto do cliente usam palavras
  diferentes para a mesma coisa.

Os filtros de metadados ("nota baixa" -> nota 1 a 2) vêm de regras, nos dois
modos (src/retrieval/filtros.py): o modelo de 7B inventava recortes.

Sem o Ollama, uma versão por regras faz o mínimo: tipo por palavras-chave e a
pergunta original mais o seu núcleo como consultas. Sem o LLM não há decisão de
escopo: a recusa depende só do limiar.
"""

import re
from dataclasses import dataclass, field

from src.generation.llm import gerar_json
from src.preprocessing.limpeza import tokenizar
from src.retrieval.filtros import Filtros, extrair_filtros
from src.retrieval.nucleo import nucleo


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


ASSUNTOS = ["experiencia_de_compra", "dados_da_empresa", "conhecimento_geral", "tarefa_de_texto"]

# Palavras que não são assunto: moldura da pergunta e termos que só dizem "o recorte".
_GENERICOS = set(
    """clientes cliente consumidores produto produtos categoria avaliacoes avaliacao compras compra
    pedidos pedido feitas feitos tiveram deram mesmo assim dizem relatam falam acham aparecem
    temas tema problemas problema reclamacoes reclamacao elogios elogiam elogio relatados relatado
    principais quais qual sao foram sobre compraram nota notas ha existem padroes padrao aspectos
    comuns frequentes identificados podem ser relacionadas relacionados experiencias experiencia
    descrevem avaliam o que""".split()
)


def _esquema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        # "motivo" antes de "assunto": o modelo gera os campos nessa ordem e classifica depois de explicar.
        "required": ["motivo", "assunto", "tipo", "tipo_tema", "consultas"],
        "properties": {
            "motivo": {"type": "string"},
            "assunto": {"type": "string", "enum": ASSUNTOS},
            "tipo": {"type": "string", "enum": ["frequencia", "exploratoria", "evidencia"]},
            "tipo_tema": {"type": "string", "enum": ["problema", "elogio", "todos"]},
            "consultas": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
        },
    }


SYSTEM = """Você prepara perguntas para um sistema de busca sobre avaliações escritas por clientes de um e-commerce brasileiro (Olist, 2016 a 2018).

Responda em JSON com:

1. motivo: uma frase dizendo sobre o que é a pergunta.

2. assunto, um destes:
   - "experiencia_de_compra": a pergunta é sobre o que clientes dizem, relatam, elogiam, reclamam ou acham de qualquer parte da compra (produto, entrega, atendimento, vendedor, pagamento, nota fiscal, embalagem, preço, frete, troca, devolução), inclusive de coisas que talvez ninguém tenha comentado. Você não decide se há avaliações sobre o assunto: a busca decide.
   - "dados_da_empresa": números ou fatos internos da empresa (faturamento, lucro, ações, funcionários, estoque).
   - "conhecimento_geral": fatos do mundo que não vêm de clientes (geografia, história, esporte, clima, receitas).
   - "tarefa_de_texto": pedido para escrever, traduzir ou resumir um texto.

   Exemplos:
   "O que os clientes dizem sobre o manual de instruções?" -> experiencia_de_compra
   "Há reclamações sobre pagamento por boleto?" -> experiencia_de_compra
   "Quantos pedidos a empresa vendeu em 2018?" -> dados_da_empresa
   "Quem descobriu o Brasil?" -> conhecimento_geral
   "Escreva um e-mail de desculpas." -> tarefa_de_texto

3. tipo:
   - "frequencia": pede ranking, "principais", "mais comuns", "com maior frequência", padrões recorrentes.
   - "evidencia": pede quais avaliações sustentam uma conclusão.
   - "exploratoria": o resto (o que relatam sobre X, como descrevem Y).

4. tipo_tema: "problema" se o foco é reclamação ou insatisfação, "elogio" se é o que valorizam ou elogiam, "todos" se ambos ou neutro.

5. consultas: 2 ou 3 frases curtas como um cliente escreveria na avaliação, com palavras diferentes para a mesma coisa. Exemplo para "problemas de entrega": "produto chegou atrasado, demorou muito", "não recebi o produto até hoje". Não repita a pergunta."""


def _ajustar(analise: "Analise", pergunta: str, categorias: list[str], filtros_interface: Filtros | None) -> "Analise":
    """Filtros por regras (src/retrieval/filtros.py) e, se depois de tirar o recorte
    não sobra assunto, a pergunta vira de frequência dentro do recorte:
    "o que os satisfeitos dizem sobre beleza e saúde" = temas mais citados nesse recorte."""
    # Contagem de temas só quando a pergunta pede ranking de forma explícita: o modelo
    # marcava "Há reclamações sobre manual de instruções?" como frequência, e a tabela
    # de temas não fala de manual.
    if analise.tipo == "frequencia" and not _FREQUENCIA.search(pergunta):
        analise.tipo = "exploratoria"
    extraidos, usados = extrair_filtros(pergunta, categorias)
    analise.filtros = filtros_interface if filtros_interface and not filtros_interface.vazio() else extraidos
    resto = [t for t in tokenizar(nucleo(pergunta)) if t not in usados and t not in _GENERICOS and not t.isdigit()]
    if not resto and not analise.filtros.vazio() and analise.tipo == "exploratoria":
        analise.tipo = "frequencia"
        analise.motivo += " Sem assunto além do recorte: respondida pela frequência de temas no recorte."
    return analise


def analisar_com_llm(pergunta: str, categorias: list[str], filtros_interface: Filtros | None = None) -> Analise:
    dados, uso = gerar_json(SYSTEM, f"Pergunta: {pergunta}", _esquema(), max_tokens=600)
    analise = Analise(
        no_escopo=dados["assunto"] == "experiencia_de_compra",
        tipo=dados["tipo"],
        tipo_tema=dados["tipo_tema"],
        consultas=[pergunta] + [c for c in dados["consultas"] if c.strip()][:3],
        filtros=Filtros(),
        motivo=f'{dados["motivo"]} (assunto: {dados["assunto"]})',
        uso=uso,
    )
    return _ajustar(analise, pergunta, categorias, filtros_interface)


_FREQUENCIA = re.compile(
    r"principa|mais comu|mais frequ|maior frequ|recorrent|padr(ao|ões|oes)|ranking|mais (elogi|reclam|relat)"
    r"|^\s*quais (s[aã]o |foram )?(os |as )?(problemas|reclama\w*|elogios|temas|aspectos)\b",
    re.I,
)
_EVIDENCIA = re.compile(r"quais avalia\w+ sustenta|evid[eê]ncia", re.I)
_ELOGIO = re.compile(r"elogi|valoriz|positiv|satisfeit|gost", re.I)
_PROBLEMA = re.compile(r"problem|reclam|insatisf|ruim|negativ|defeit|atras", re.I)


def analisar_por_regras(pergunta: str, filtros: Filtros | None = None, categorias: list[str] | None = None) -> Analise:
    tipo = "evidencia" if _EVIDENCIA.search(pergunta) else "frequencia" if _FREQUENCIA.search(pergunta) else "exploratoria"
    tipo_tema = "elogio" if _ELOGIO.search(pergunta) and not _PROBLEMA.search(pergunta) else "problema" if _PROBLEMA.search(pergunta) else "todos"
    consultas = [pergunta] + ([nucleo(pergunta)] if nucleo(pergunta) != pergunta.strip().rstrip("?. ") else [])
    analise = Analise(True, tipo, tipo_tema, consultas, Filtros(), "Análise por regras (sem LLM).", origem="regras")
    return _ajustar(analise, pergunta, categorias or [], filtros)
