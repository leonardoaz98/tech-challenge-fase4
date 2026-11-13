"""Construção do contexto, geração da resposta e verificação das citações.

O contexto enviado ao modelo tem três partes:
1. A pergunta e os filtros aplicados.
2. Para perguntas de frequência: a contagem de temas na base inteira filtrada
   (é a ÚNICA fonte de números que o modelo pode usar).
3. As avaliações recuperadas, numeradas [R1]..[Rn], com nota, categoria e entrega.

Regras do prompt: usar só o contexto, citar [Rn] em cada afirmação, não inventar
número, dizer quando a evidência não basta. Depois da geração, o código confere
cada citação: [Rn] fora do contexto é removido e contado como citação inválida.
"""

import re
from dataclasses import dataclass, field

from src.generation.llm import gerar_json
from src.retrieval.busca import Evidencia

STATUS_LEGIVEL = {"no_prazo": "entregue no prazo", "atrasado": "entregue com atraso", "nao_entregue": "não entregue"}

SYSTEM = """Você é um analista de Voice of Customer. Responde perguntas de gestores usando SOMENTE as avaliações de clientes e as contagens fornecidas no contexto.

Regras:
1. Toda afirmação sobre o que clientes disseram leva a citação das avaliações que a sustentam, no formato [R3] ou [R3][R7], logo após a frase.
2. Números (quantidades, percentuais, rankings) só podem vir da tabela "Frequência de temas na base". Nunca conte as avaliações do contexto para inferir frequência: elas foram selecionadas por relevância, não são amostra representativa. Se não houver tabela, não afirme frequência; descreva o que aparece nas evidências.
3. Não acrescente conhecimento externo, causas não relatadas ou recomendações que as evidências não sustentem. Interpretar é permitido quando a interpretação está marcada como tal e apoiada em citações.
4. Se as evidências não respondem a pergunta, ou respondem só em parte, diga isso claramente em "suficiente" (false) e em "lacunas". Uma resposta curta e honesta vale mais que uma resposta completa sem suporte.
5. Escreva em português, para um gestor: direto, em tópicos quando ajudar, sem jargão técnico.
6. Cada insight em "insights" é uma conclusão curta com a lista das avaliações (ex.: ["R1", "R4"]) que a sustentam."""

ESQUEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["suficiente", "resposta", "insights", "lacunas"],
    "properties": {
        "suficiente": {"type": "boolean"},
        "resposta": {"type": "string"},
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["insight", "evidencias"],
                "properties": {"insight": {"type": "string"}, "evidencias": {"type": "array", "items": {"type": "string"}}},
            },
        },
        "lacunas": {"type": "string"},
    },
}


@dataclass
class Resposta:
    texto: str
    suficiente: bool
    insights: list[dict]
    lacunas: str
    citacoes_validas: int = 0
    citacoes_invalidas: list[str] = field(default_factory=list)
    contexto: str = ""
    uso: dict = field(default_factory=dict)


def montar_contexto(pergunta: str, filtros_desc: str, evidencias: list[Evidencia], frequencias: dict | None) -> str:
    partes = [f"Pergunta do gestor: {pergunta}", f"Filtros aplicados: {filtros_desc}"]
    if frequencias:
        linhas = [f"Frequência de temas na base (universo: {frequencias['universo']} avaliações com texto após filtros):"]
        for t in frequencias["temas"]:
            linhas.append(f"- {t['rotulo']} ({t['tipo']}): {t['avaliacoes']} avaliações ({t['percentual']}%)")
        linhas.append("Uma avaliação pode ter mais de um tema. Temas detectados por regras de texto (ver limitações).")
        partes.append("\n".join(linhas))
    linhas = ["Avaliações recuperadas:"]
    for i, e in enumerate(evidencias, start=1):
        linhas.append(
            f"[R{i}] nota {e.nota} | {e.categoria} | {STATUS_LEGIVEL.get(e.status_entrega, e.status_entrega)} | compra {e.data_compra}\n\"{e.texto}\""
        )
    partes.append("\n".join(linhas))
    return "\n\n".join(partes)


_CITACAO = re.compile(r"\[R(\d+)\]")


def verificar_citacoes(texto: str, n: int) -> tuple[str, int, list[str]]:
    validas, invalidas = 0, []

    def trocar(m: re.Match) -> str:
        nonlocal validas
        if 1 <= int(m.group(1)) <= n:
            validas += 1
            return m.group(0)
        invalidas.append(m.group(0))
        return ""

    return _CITACAO.sub(trocar, texto), validas, invalidas


def gerar(pergunta: str, filtros_desc: str, evidencias: list[Evidencia], frequencias: dict | None) -> Resposta:
    contexto = montar_contexto(pergunta, filtros_desc, evidencias, frequencias)
    dados, uso = gerar_json(SYSTEM, contexto, ESQUEMA, esforco="medium")
    texto, validas, invalidas = verificar_citacoes(dados["resposta"], len(evidencias))
    rotulos_ok = {f"R{i}" for i in range(1, len(evidencias) + 1)}
    insights = []
    for ins in dados["insights"]:
        refs = [r.strip("[]") for r in ins["evidencias"]]
        invalidas += [r for r in refs if r not in rotulos_ok]
        insights.append({"insight": ins["insight"], "evidencias": [r for r in refs if r in rotulos_ok]})
    return Resposta(texto, dados["suficiente"], insights, dados["lacunas"], validas, invalidas, contexto, uso)
