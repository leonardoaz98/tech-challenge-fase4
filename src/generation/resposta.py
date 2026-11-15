"""Construção do contexto, geração da resposta e verificação das citações.

O contexto enviado ao modelo tem três partes:
1. A pergunta e os filtros aplicados.
2. Para perguntas de frequência: a contagem de temas na base inteira filtrada,
   para o modelo saber quais temas descrever (os números ele não escreve).
3. As avaliações recuperadas, numeradas [R1]..[Rn], com nota, categoria e entrega.

A resposta é estruturada: cada insight vem com a lista das avaliações que o
sustentam, e o esquema JSON só aceita rótulos que existem no contexto (R1..Rn).
O texto final é montado pelo código a partir dos insights, então toda conclusão
sai citada. Depois da geração, o código ainda confere:
- citações [Rn] no resumo fora do contexto: removidas e contadas;
- percentuais escritos pelo modelo: removidos e contados. Os números da resposta
  (ranking de temas) são escritos pelo código a partir da contagem; um modelo
  pequeno tende a somar, arredondar ou dizer "a maioria" de 32%.
"""

import re
from dataclasses import dataclass, field

from src import config
from src.generation.llm import gerar_json
from src.retrieval.busca import Evidencia

STATUS_LEGIVEL = {"no_prazo": "entregue no prazo", "atrasado": "entregue com atraso", "nao_entregue": "não entregue"}

SYSTEM = """Você é um analista que lê avaliações de clientes e responde a um gestor. Use SOMENTE o contexto recebido.

Responda em JSON, nesta ordem:

1. assunto: o assunto específico da pergunta, em poucas palavras (ex.: "nota fiscal", "restaurante da loja", "problemas com nota baixa").
2. avaliacoes_sobre_o_assunto: os rótulos das avaliações que falam EXPLICITAMENTE desse assunto. Uma avaliação que só elogia ou critica a compra em geral, sem falar do assunto, não entra. Pode ficar vazia.
3. insights: até 5 conclusões curtas que respondem à pergunta, cada uma com "evidencias": os rótulos das avaliações que dizem aquilo. Use só avaliações da lista do item 2. Se a lista do item 2 estiver vazia, deixe insights vazio.
4. resumo: uma ou duas frases que respondem à pergunta de forma direta.
5. lacunas: o que as avaliações não permitem afirmar. Escreva "" se não houver.

Regras:
- Não escreva números, percentuais nem quantificadores ("a maioria", "muitos", "poucos"). Quando houver tabela de frequência, ela é mostrada ao gestor junto com a sua resposta: descreva o que os clientes dizem em cada tema, não quanto.
- Não use conhecimento externo nem invente causas que os clientes não relataram.
- Português simples, para um gestor."""


def _esquema(n_evidencias: int) -> dict:
    """Esquema por chamada: as citações só podem ser rótulos que existem no contexto."""
    rotulo = {"type": "string", "enum": [f"R{i}" for i in range(1, n_evidencias + 1)]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assunto", "avaliacoes_sobre_o_assunto", "insights", "resumo", "lacunas"],
        "properties": {
            "assunto": {"type": "string"},
            "avaliacoes_sobre_o_assunto": {"type": "array", "items": rotulo},
            "insights": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["insight", "evidencias"],
                    "properties": {"insight": {"type": "string"}, "evidencias": {"type": "array", "minItems": 1, "items": rotulo}},
                },
            },
            "resumo": {"type": "string"},
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
    numeros_removidos: list[str] = field(default_factory=list)
    sobre_o_assunto: list[str] = field(default_factory=list)
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


_PERCENTUAL = re.compile(r"\b\d+(?:[.,]\d+)?\s?%")


def verificar_numeros(texto: str, permitidos: set[str]) -> tuple[str, list[str]]:
    """Remove percentuais que não estão na tabela de temas (o modelo somou ou inventou)."""
    removidos = []

    def trocar(m: re.Match) -> str:
        valor = m.group(0).replace(",", ".").replace(" ", "")
        if valor in permitidos:
            return m.group(0)
        removidos.append(m.group(0))
        return "[número removido: não está na tabela]"

    return _PERCENTUAL.sub(trocar, texto), removidos


def ranking_temas(frequencias: dict, n: int = 5) -> str:
    """Os números da resposta vêm daqui, escritos pelo código a partir da contagem."""
    temas = [t for t in frequencias["temas"] if t["avaliacoes"] > 0][:n]
    linhas = [f"{i}. {t['rotulo']}: {t['avaliacoes']} avaliações ({str(t['percentual']).replace('.', ',')}%)" for i, t in enumerate(temas, 1)]
    universo = f"{frequencias['universo']:,}".replace(",", ".")
    return f"Temas mais citados entre {universo} avaliações com texto (contagem na base, não só nas recuperadas):\n" + "\n".join(linhas)


def gerar(pergunta: str, filtros_desc: str, evidencias: list[Evidencia], frequencias: dict | None) -> Resposta:
    contexto = montar_contexto(pergunta, filtros_desc, evidencias, frequencias)
    dados, uso = gerar_json(SYSTEM, contexto, _esquema(len(evidencias)))
    rotulos_ok = {f"R{i}" for i in range(1, len(evidencias) + 1)}
    sobre = [r for r in dict.fromkeys(dados["avaliacoes_sobre_o_assunto"]) if r in rotulos_ok]

    resumo, validas, invalidas = verificar_citacoes(dados["resumo"], len(evidencias))
    resumo, removidos = verificar_numeros(resumo, set())
    insights, linhas = [], []
    for ins in dados["insights"]:
        refs = list(dict.fromkeys(r.strip("[]") for r in ins["evidencias"]))
        invalidas += [r for r in refs if r not in rotulos_ok]
        refs = [r for r in refs if r in rotulos_ok]
        texto_insight, rem = verificar_numeros(ins["insight"], set())
        removidos += rem
        if refs:  # insight sem evidência válida não entra na resposta
            validas += len(refs)
            insights.append({"insight": texto_insight, "evidencias": refs})
            linhas.append(f"- {texto_insight} " + "".join(f"[{r}]" for r in refs))

    # Suficiência decidida pelo código: perguntas de frequência se apoiam na contagem;
    # as demais precisam de MIN_EVIDENCIAS avaliações que o modelo apontou como sendo
    # sobre o assunto (filtra o que o reranker deixou passar, como "recomendo a loja"
    # numa pergunta sobre o restaurante da loja).
    suficiente = bool(insights) and (frequencias is not None or len(sobre) >= config.MIN_EVIDENCIAS)
    partes = [resumo.strip()]
    if frequencias:
        partes.append(ranking_temas(frequencias))
    if linhas:
        partes.append("\n".join(linhas))
    texto = "\n\n".join(p for p in partes if p)
    return Resposta(texto, suficiente, insights, dados["lacunas"], validas, invalidas, removidos, sobre, contexto, uso)
