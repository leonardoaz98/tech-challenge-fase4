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


def nucleo(pergunta: str) -> str:
    base = pergunta.strip().rstrip("?. ")
    n = _MOLDURA.sub("", base, count=1).strip()
    return n if n and n != base else base
