# expense-classifier

Classificador de despesas pessoais por descrição de transação
bancária (formatos Banco do Brasil e Nubank). Sem API externa, seus dados
não saem da máquina.

## Setup (uv)

Este projeto usa [uv](https://docs.astral.sh/uv/). Requer Python ≥ 3.10
(o repositório fixa 3.12 via `uv.lock`). Dependências de runtime:
`pandas`, `scikit-learn`, `joblib`.

```bash
uv sync --extra dev     # cria a .venv e instala runtime + dev (pytest, ruff)
# ou apenas runtime:
uv sync
```

`uv sync` lê o `uv.lock`, cria a `.venv/` e instala o pacote em modo editável.

<details>
<summary>Sem uv? (pip clássico)</summary>

```bash
pip install -e ".[dev]"     # dev: pytest + ruff
# ou apenas runtime:
pip install .
```
</details>

## Uso (CLI)

Prefixe os comandos com `uv run` (não precisa ativar a venv manualmente).
Alternativa: `source .venv/bin/activate` uma vez e depois chamar
`expense-classifier ...` direto.

```bash
uv run expense-classifier classify extrato.csv -o classificado.csv
uv run expense-classifier report classificado.csv
uv run expense-classifier train                      # retreina após editar labels.csv
```

### Opções globais

Vêm **antes** do subcomando:

| opção            | efeito                                                                 |
| ---------------- | ---------------------------------------------------------------------- |
| `--data-dir DIR` | onde ficam `merchants.json`/`labels.csv`/`model.joblib` (padrão: `$EXPENSE_CLASSIFIER_HOME` ou `~/.config/expense-classifier`) |
| `-v, --verbose`  | logs em nível DEBUG                                                     |
| `--version`      | imprime a versão e sai                                                  |

```bash
uv run expense-classifier --data-dir ./meus-dados classify extrato.csv
```

### `classify` — classifica um CSV de transações

```bash
uv run expense-classifier classify extrato.csv                    # -> expenses_classified.csv
uv run expense-classifier classify extrato.csv -o classificado.csv
uv run expense-classifier classify extrato.csv --no-retrain       # usa o model.joblib salvo
```

O CSV de entrada precisa da coluna `description`; `transaction_type` e
`amount` são usados se existirem. A saída preserva todas as colunas originais
e adiciona `description_normalized`, `category`, `method` e `confidence`.

Por padrão o modelo ML é **retreinado** com o próprio arquivo de entrada
(bootstrap pelas regras) antes de classificar; `--no-retrain` pula essa etapa
e usa o `model.joblib` existente. Ao final o comando imprime um resumo
(classificadas / ignoradas / para revisar, contagem por método) e lista as
descrições que caíram em `revisar` — candidatas a entrar no `merchants.json`
ou no `labels.csv`:

```
[INFO] modelo treinado: 103 exemplos, 9 classes -> ~/.config/expense-classifier/model.joblib

95 transações -> classificado.csv
  classificadas : 51
  ignoradas     : 14 (fatura/receita/estorno)
  para revisar  : 30

por método:
method
merchant    50
none        30
ignore      14
keyword      1

Para revisão (edite merchants.json ou labels.csv):
description_normalized
HOT DOG DUPIRU          3
PIX MARKETPLACE         1
...
```

### `train` — (re)treina o modelo ML

```bash
uv run expense-classifier train                # regras + labels.csv
uv run expense-classifier train extrato.csv    # + bootstrap com um CSV extra
```

Rode após editar `merchants.json` ou `labels.csv`. O CSV é opcional: se
passado, as regras rotulam o que conseguem dele e esses exemplos entram no
treino. Sai com código `1` se não houver dados suficientes para treinar.

### `report` — resumo por categoria

```bash
uv run expense-classifier report classificado.csv
```

Recebe um CSV **já classificado** (precisa das colunas `category` e `amount`),
exclui `ignorado`/`revisar` e imprime total, número de transações e ticket
médio por categoria, ordenado por gasto:

```
                         total  transacoes  ticket_medio
category
moradia                3903.58           5        780.72
saúde                  3345.72           6        557.62
educação               1441.40           5        288.28
supermercado           1055.06          24         43.96
bares e restaurantes    433.94           5         86.79
...

Total classificado: R$ 10,571.20
```

Como biblioteca: veja a seção [Library API](#library-api) abaixo.

## Library API

O layout é plano (módulos no topo, sem pacote); importe de cada módulo:

```python
from cascade import Classifier, Classification   # orquestra a cascata; dataclass do resultado
from config import Settings, CATEGORIES, REVIEW_LABEL, IGNORE_LABEL
from rules import MerchantsFileError             # merchants.json malformado ou categoria inválida
from normalize import normalize                  # descrição crua -> merchant limpo
```

Para que esses `import` de nível superior funcionem, a raiz do repositório
precisa estar no `sys.path` — o que já acontece após `uv sync` (instala os
módulos em modo editável) ou ao rodar a partir da própria pasta do projeto.

> Todos os exemplos abaixo foram executados contra o pacote instalado —
> as saídas nos comentários são reais.

### 1. Classificar uma descrição

```python
from cascade import Classifier
from config import Settings

clf = Classifier(Settings())   # primeiro uso semeia merchants.json no data dir

clf.classify_one("PIX - ENVIADO 20/06 17:23 GIASSI & CIA LTDA")
# Classification(category='supermercado', method='merchant', confidence=1.0)

clf.classify_one("Transferência enviada|FULANO DE TAL")
# Classification(category='revisar', method='none', confidence=0.0)
#   ^ pessoa física: a guarda impede o ML de chutar

clf.classify_one("Pagamento de fatura")
# Classification(category='ignorado', method='ignore', confidence=1.0)
```

`Classification` é um frozen dataclass — seguro para usar como chave de
dict, em sets e em código concorrente:

```python
r = clf.classify_one("PAGTO UNIMED    UNIMED FLORIANÓPOLIS")
r.category     # 'saúde'
r.method       # 'merchant'  (ignore | merchant | keyword | ml | none)
r.confidence   # 1.0
```

### 2. Classificar um DataFrame

```python
import pandas as pd
from cascade import Classifier
from config import Settings

df = pd.read_csv("extrato.csv")   # exige coluna 'description';
                                  # 'transaction_type' é usada se existir

clf = Classifier(Settings())
out = clf.classify_dataframe(df)  # preserva todas as colunas originais

out[["description_normalized", "category", "method", "confidence"]]
# description_normalized  category   method  confidence
#   UNIMED FLORIANOPOLIS     saúde  merchant        1.0
#  FARMACIA CENTRAL LTDA     saúde  merchant        1.0
```

Sem a coluna `description`, `classify_dataframe` levanta `ValueError` com a
lista de colunas presentes.

### 3. Retreinar o ML no ato

Passe o DataFrame em `retrain_with` para fazer o bootstrap (regras rotulam o
que conseguem e viram exemplos de treino) e persistir o modelo em uma única
construção:

```python
clf = Classifier(Settings(), retrain_with=df)   # treina + salva model.joblib
clf.model is not None                           # True se houve dados suficientes
```

Sem `retrain_with`, o construtor carrega o `model.joblib` existente; se o
arquivo não existir ou for incompatível (versão de sklearn diferente), degrada
para regras-apenas com um warning — nunca levanta exceção por isso.

### 4. Diretório de dados customizado

Útil para testes, multiusuário ou empacotar num serviço:

```python
from pathlib import Path
from cascade import Classifier
from config import Settings

settings = Settings(data_dir=Path("/srv/app/classifier-data"))
clf = Classifier(settings)          # ensure_initialized() roda no construtor

settings.merchants_path   # /srv/app/classifier-data/merchants.json
settings.labels_path      # /srv/app/classifier-data/labels.csv
settings.model_path       # /srv/app/classifier-data/model.joblib
```

Precedência: `Settings(data_dir=...)` > `$EXPENSE_CLASSIFIER_HOME` >
`~/.config/expense-classifier/`.

### 5. Tratamento de erros

```python
from cascade import Classifier
from config import Settings
from rules import MerchantsFileError

try:
    clf = Classifier(Settings())
except MerchantsFileError as exc:
    # JSON malformado OU categoria fora de CATEGORIES —
    # a mensagem inclui a chave problemática e a lista de categorias válidas
    print(f"corrija seu merchants.json: {exc}")
```

### 6. Só a normalização

`normalize()` é pura e sem estado — dá para usar isolada em pipelines:

```python
from normalize import normalize

normalize("Transferência enviada|GIASSI &amp; CIA LTDA")
# 'GIASSI & CIA LTDA'

normalize("PIX - ENVIADO   03/06 19:54 ENIR LUCIA TIDRE 01776154")
# 'ENIR LUCIA TIDRE'

normalize(float("nan"))   # células NaN de pandas viram string vazia
# ''
```

### 7. Exemplo integrado: resumo de gastos

```python
import pandas as pd
from cascade import Classifier
from config import IGNORE_LABEL, REVIEW_LABEL, Settings

clf = Classifier(Settings())
out = clf.classify_dataframe(pd.read_csv("extrato.csv"))

spend = out[~out["category"].isin([IGNORE_LABEL, REVIEW_LABEL])]
print(spend.groupby("category")["amount"].sum().abs().sort_values(ascending=False))

pendentes = out.loc[out["category"] == REVIEW_LABEL, "description_normalized"]
print(pendentes.value_counts())   # candidatos a entrar no merchants.json
```

### 8. Exemplo: endpoint Django/DRF

O `Classifier` é seguro para reuso entre requests (regras e modelo são
carregados uma vez e apenas lidos depois). Instancie no nível do módulo:

```python
# app/services/classifier.py
from cascade import Classifier
from config import Settings

_classifier = Classifier(Settings())   # singleton por processo

def classify_description(description: str, transaction_type: str | None = None):
    r = _classifier.classify_one(description, transaction_type)
    return {"category": r.category, "method": r.method, "confidence": r.confidence}
```

```python
# app/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from app.services.classifier import classify_description

@api_view(["POST"])
def classify(request):
    return Response(classify_description(
        request.data["description"],
        request.data.get("transaction_type"),
    ))
```

Nota: o snippet Django/DRF acima é ilustrativo (não faz parte do pacote nem
foi executado nos testes); o restante dos exemplos desta seção foi verificado
contra o pacote instalado.

## Arquitetura

```
descrição crua
   │ normalize() — prefixos de banco, HTML entities, acentos, CNPJ/dígitos
   ▼
1. IGNORE    fatura, receita, estorno, IOF de volta       → "ignorado"
2. MERCHANT  substring lookup em merchants.json           (confiança 1.0)
3. KEYWORD   FARMACIA, POSTO, SORVETERIA, ...             (0.9)
4. GUARDA    nome de pessoa física? ML nunca chuta        → "revisar"
5. ML        TF-IDF char_wb(2,5) + LinearSVC calibrado    (aceita se ≥ 0.55)
6. REVISAR   resto vai para revisão humana
```

**Por que não ML puro?** Extratos não vêm rotulados — não há ground truth.
O bootstrap usa as regras como professor; o ML generaliza variações que as
regras não cobrem (nomes truncados pelo banco, grafias alternativas). PIX
para pessoa física não carrega sinal textual: chutar seria pior que perguntar.

## Dados do usuário

Ficam em `~/.config/expense-classifier/` (ou `$EXPENSE_CLASSIFIER_HOME`,
ou `--data-dir`):

| arquivo         | papel                                                    |
| --------------- | -------------------------------------------------------- |
| `merchants.json`| fonte da verdade: `"TRECHO DO NOME": "categoria"`        |
| `labels.csv`    | correções avulsas (`description,category`) p/ treino ML  |
| `model.joblib`  | modelo treinado (escrita atômica; regenerável)           |

No primeiro uso, `merchants.json` é semeado a partir do default embarcado.

## Loop de melhoria

1. Rode `classify` — o que cair em `revisar` é listado ao final.
2. Merchants/pessoas recorrentes → `merchants.json`.
3. Casos avulsos → `labels.csv` + `train`.
4. Repita. A cobertura converge rápido: seus merchants recorrentes dominam
   o volume.

## Desenvolvimento

```bash
uv run pytest            # 37 testes
uv run ruff check src tests
```

## Categorias válidas

supermercado · bares e restaurantes · cafés e panificadoras · moradia ·
transporte · lazer · vestuário · saúde · serviços · pet · educação
(+ pseudo-categorias de sistema: `ignorado`, `revisar`)

## Limites conhecidos

- PIX para pessoa física é indecidível pelo texto → sempre `revisar` até você
  fixar a pessoa em `merchants.json`.
- Intermediários de pagamento (SHPP/Shopee, marketplaces) mascaram a categoria
  real da compra.
- O threshold do ML (0.55) é conservador de propósito; com mais labels dá para
  calibrar via validação cruzada.
- `model.joblib` é acoplado à versão do scikit-learn; se falhar ao carregar,
  o load degrada para regras e avisa para retreinar.
