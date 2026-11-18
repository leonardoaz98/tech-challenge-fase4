"""Núcleo da pergunta: o assunto, sem a moldura de pergunta de gestor.

"O que os clientes falam sobre a nota fiscal?" -> "a nota fiscal".

O cross-encoder compara a consulta com cada avaliação. Avaliações têm de 3 a ~40
palavras e não repetem a moldura "o que os clientes relatam sobre", então o
enunciado inteiro derruba o score de avaliações claramente relevantes
(ex.: "Faltou a nota fiscal" ficava em -3,3). Com o núcleo, o score mede se a
avaliação fala do assunto, que é o que o limiar de recusa precisa saber.
A comparação está em reports/calibracao_limiar.json.
"""

import re

_MOLDURA = re.compile(
    r"^\s*(o que|como|quais?|que|h[aá]|existem|os clientes|qual)\b.*?"
    r"\b(sobre|com|de|d[oa]s?|a[os]?|em|quando|relacionad[oa]s? a|que)\s+",
    re.I,
)


# Pedido de evidência: "Quais avaliações sustentam a conclusão de que existem problemas
# recorrentes relacionados ao prazo de entrega?" -> "problemas relacionados ao prazo de
# entrega". Sem isso, a moldura genérica parava em "sustentam a" e o núcleo virava
# "conclusão de que existem...", que nenhuma avaliação parece com.
_EVIDENCIA = re.compile(
    r"^\s*quais?\s+(avalia\w*|coment\w*|relatos?)\s+"
    r"(sustenta\w*|comprova\w*|mostra\w*|confirma\w*|indica\w*|apoia\w*|evidencia\w*)\s+"
    r"(a\s+(conclus\w+|ideia|hip[oó]tese)\s+)?(de\s+)?que\s+((existem|h[aá])\s+)?",
    re.I,
)
_FREQUENCIA = re.compile(r"\s*\b(recorrentes?|frequentes?)\b", re.I)


def nucleo(pergunta: str) -> str:
    base = pergunta.strip().rstrip("?. ")
    if _EVIDENCIA.match(base):
        n = _FREQUENCIA.sub("", _EVIDENCIA.sub("", base, count=1)).strip()
        if n:
            return n
    n = _MOLDURA.sub("", base, count=1).strip()
    return n if n and n != base else base
