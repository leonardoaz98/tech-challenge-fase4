"""Classificação temática de TODA a base, para responder perguntas de frequência.

Por que existe: "quais problemas aparecem com maior frequência?" é uma pergunta de
contagem. A busca devolve as 15 avaliações mais parecidas com a pergunta, não uma
amostra representativa; contar temas dentro delas inventaria a frequência. Aqui cada
avaliação recebe zero ou mais temas e a contagem é feita na base inteira.

Método: dicionário de expressões regulares sobre o texto normalizado (sem acento,
minúsculo). Escolhido por ser transparente (cada tema é auditável linha a linha),
gratuito e determinístico. A qualidade é medida em src/evaluation/avaliar_temas.py
contra uma amostra rotulada, e o resultado vai para reports/temas_validacao.json.

Uso: python -m src.temas.temas
"""

import json
import re
from dataclasses import dataclass

import pandas as pd

from src import config
from src.logger import obter_logger
from src.retrieval.filtros import Filtros, mascara

log = obter_logger(__name__)


@dataclass(frozen=True)
class Tema:
    nome: str
    rotulo: str
    tipo: str  # "problema" | "elogio" | "neutro"
    padrao: str
    nega: bool = False  # descarta a ocorrência se houver "nao" logo antes


TEMAS: list[Tema] = [
    Tema("atraso", "Atraso na entrega", "problema",
         r"atras|demor|fora do prazo|passou do prazo|apos o prazo|depois do prazo|prazo (ja )?(expirou|vencido|venceu|estourad|esgotad)"
         r"|ainda nao (chegou|recebi)|nao chegou (no|dentro do) prazo|prazo nao (foi )?cumprid|nao cumpr\w+ (o )?prazo"),
    Tema("nao_entregue", "Produto não entregue", "problema",
         r"nao (recebi|recebemos|chegou|chegaram|foi entregue|foram entregues|me entregaram|entregaram|veio)"
         r"|nao (foi|foram) entregue|nao entregue|nao recebid|nunca (chegou|recebi)|ainda aguardando|aguardando (a )?(entrega|o produto|chegar|meu|minha)"
         r"|no aguardo|aguardo (o|a|do|da|meu|minha)|cade (o|meu|minha)"),
    Tema("incompleto", "Pedido incompleto", "problema",
         r"falt(ou|ando|a|aram) (um|uma|o|a|os|as|\d|item|itens|peca|pecas|produto|parte)|incomplet"
         r"|(veio|recebi|chegou|chegaram) (so|somente|apenas) (um|uma|\d|parte|o|a)|so (veio|chegou|recebi) (um|uma|\d)"
         r"|de (dois|duas|tres|\d) (produtos|itens|unidades)? ?(so|somente|apenas)"
         r"|(entreg\w*|enviad\w*|mandaram|recebid\w*) (so|somente|apenas) (\d+|um|uma|o|a|parte|metade)"
         r"|(so|somente|apenas) (\d+|um|uma) (foi|foram|veio|chegou) "),
    Tema("produto_errado", "Produto errado ou diferente do anunciado", "problema",
         r"(produto|cor|tamanho|modelo|item|numero|voltagem) (errad|diferente|trocad|incorret)|veio (errad|diferente|outr|trocad)"
         r"|diferente d[ao] (anunci|foto|descri|que (comprei|pedi))|nao (e|era|corresponde)( o| a)? (que|ao|a|com) (comprei|pedi|anunci|foto|descri)"
         r"|falsificad|nao (e|era) original|nao original|pirata|replica|remanufaturad"),
    Tema("defeito_avaria", "Defeito ou avaria", "problema",
         r"defeit|quebrad|danificad|avariad|amassad|riscad|trincad|rachad|estragad|nao funciona|nao (esta|estava|estao) funcionando|nao funcionou|parou de funcionar|nao liga\b"
         r"|veio (aberto|violad|sujo|usad)|com problema|vazand|furad|rasgad"),
    Tema("qualidade_ruim", "Qualidade ruim", "problema",
         r"(qualidade|material|acabamento) (ruim|pessim|baix|inferior|fraca|fraco|horrivel|duvidos|deixa a desejar)"
         r"|(ma|baixa|pessima|horrivel) qualidade|fragil|frageis|descasc|desbot|soltou|nao vale (o|a pena)|nao gostei"),
    Tema("atendimento_ruim", "Atendimento e falta de retorno", "problema",
         r"nao (respond|retorn|deram retorno|tive retorno|obtive|tive resposta|me respond|resolve)|sem (resposta|retorno|solucao)"
         r"|ninguem (respond|resolv|retorn)|descaso|(entrei|entramos) em contato|varios contatos|ja (reclamei|liguei|mandei)"
         r"|atendimento (ruim|pessim|horrivel|demorad|fraco)|pessimo atendimento|falta de (respeito|compromisso|comunicacao)"),
    Tema("devolucao_reembolso", "Devolução, troca ou reembolso", "problema",
         r"devol|reembols|estorn|dinheiro de volta|meu dinheiro|cancel|\btroca\b|\btrocar\b|ressarc"),
    Tema("embalagem", "Embalagem", "neutro",
         r"embalag|embalad|caixa (amassad|rasgad|aberta|violad|danificad)|plastico bolha"),
    Tema("frete_preco", "Preço ou frete", "neutro",
         r"preco|barat|\bcaro\b|\bcara\b|custo beneficio|frete|valor (do produto|pago|justo)"),
    Tema("entrega_rapida", "Entrega rápida ou no prazo", "elogio",
         r"(entrega|chegou|chegaram|entregue|entregaram|recebi)( \w+){0,2} (rapid|antes|cedo|adiantad|no prazo|dentro do prazo|certinh)"
         r"|antes do prazo|antes da data|dentro do prazo|no prazo|super rapid|muito rapid|bem rapid|rapidez|agil|agilidade",
         nega=True),
    Tema("produto_bom", "Produto bom ou como anunciado", "elogio",
         r"(otim|excelent|bo[am]|perfeit|lind|maravilhos|bonit|top)\w* (produto|qualidade|material|acabamento)"
         r"|produto (e |muito |super |de )?(otim|excelent|bom|boa|perfeit|lind|maravilhos|conforme|igual|de qualidade|top|bonit)"
         r"|conforme (o |a )?(anunci|descri|foto|esperad|combinad)|igual (a|ao|as|aos) (foto|anunci|descri)|qualidade (otima|excelente|boa|muito boa|superou)"
         r"|(gostei|amei|adorei)|atendeu (as |minhas )?(expectativas|perfeitamente)|superou",
         nega=True),
    Tema("recomenda", "Recomenda a loja ou compraria de novo", "elogio",
         r"recomendo|indico|compraria (de novo|novamente)|comprarei (de novo|novamente|mais)|voltarei a comprar|comprar (de novo|novamente|mais vezes)"
         r"|parabens|confiavel|confianca|(loja|vendedor|empresa|lannister)( \w+)? (otim|excelent|confiavel|seria|atencios|nota 10|de confianca)",
         nega=True),
    Tema("atendimento_bom", "Bom atendimento", "elogio",
         r"(atendimento|vendedor|loja|suporte)( \w+)? (otimo|excelente|atencios|prestativ|eficiente|rapido|nota 10)"
         r"|(otimo|excelente|bom) atendimento|resolveram|resolveu (o problema|rapidamente)|me atenderam",
         nega=True),
]  # fmt: skip

_COMPILADOS = {t.nome: re.compile(t.padrao) for t in TEMAS}
_NEGACAO = re.compile(r"\b(nao|nem|nunca|sem)\b(\s+\w+){0,2}\s*$")


def classificar_texto(texto_busca: str) -> list[str]:
    encontrados = []
    for tema in TEMAS:
        for m in _COMPILADOS[tema.nome].finditer(texto_busca):
            if tema.nega and _NEGACAO.search(texto_busca[max(0, m.start() - 25) : m.start()]):
                continue
            encontrados.append(tema.nome)
            break
    return encontrados


def classificar_base() -> pd.DataFrame:
    docs = pd.read_parquet(config.DOCUMENTOS, columns=["review_id", "texto_busca", "nota"])
    temas = pd.DataFrame({"review_id": docs["review_id"]})
    rotulos = docs["texto_busca"].map(classificar_texto)
    for t in TEMAS:
        temas[t.nome] = rotulos.map(lambda r, n=t.nome: n in r)
    temas.to_parquet(config.TEMAS, index=False)

    cobertura = rotulos.map(len).gt(0).mean()
    resumo = {
        "documentos": len(docs),
        "com_pelo_menos_um_tema": round(float(cobertura), 3),
        "por_tema": {t.nome: int(temas[t.nome].sum()) for t in TEMAS},
    }
    (config.RELATORIOS / "temas_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), "utf-8")
    log.info("Temas: %.1f%% das avaliações com pelo menos um tema", 100 * cobertura)
    return temas


_CACHE: dict = {}


def contar(filtros: Filtros | None = None, tipo: str | None = None) -> dict:
    """Frequência de cada tema no subconjunto filtrado da base inteira."""
    if "base" not in _CACHE:
        docs = pd.read_parquet(config.DOCUMENTOS)
        _CACHE["base"] = docs.merge(pd.read_parquet(config.TEMAS), on="review_id")
    base = _CACHE["base"]
    sub = base[mascara(base, filtros or Filtros())]
    total = len(sub)
    linhas = []
    for t in TEMAS:
        if tipo and t.tipo != tipo:
            continue
        n = int(sub[t.nome].sum())
        linhas.append({"tema": t.nome, "rotulo": t.rotulo, "tipo": t.tipo, "avaliacoes": n,
                       "percentual": round(100 * n / total, 1) if total else 0.0})  # fmt: skip
    linhas.sort(key=lambda r: r["avaliacoes"], reverse=True)
    return {"universo": total, "temas": linhas}


if __name__ == "__main__":
    classificar_base()
