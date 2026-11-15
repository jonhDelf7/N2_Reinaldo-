# Atividade Prática: Análise de Dados e Machine Learning

## Estudo de caso: índice de pobreza no Brasil (08/2012 a 09/2022)

**Integrantes**

- Gustavo Samuel da Cruz Morais
- Abner Luiz Pascoal de Oliveira
- João Pedro Delfino Teixeira
- João Gabriel Arantes Sena

---

## 1. Apresentação

Neste trabalho, nosso grupo analisa o comportamento histórico da pobreza e da extrema
pobreza no Brasil entre agosto de 2012 e setembro de 2022. Para fazer essa investigação,
selecionamos uma base de dados pública mensal com 122 registros consolidados.

A nossa proposta central é cruzar o crescimento demográfico estimado com o avanço do número
de pessoas e famílias em vulnerabilidade social ao longo dessa década.

A partir da aplicação de técnicas de estatística descritiva e análise gráfica, queremos
observar tendências, identificar momentos de agravamento econômico e compreender o real peso
proporcional de cada faixa na população do país. Na segunda etapa, aplicamos dois modelos de
Machine Learning para projetar o comportamento da série nos próximos anos.

---

## 2. A base de dados

Arquivo: `indices.csv` (122 linhas, 12 colunas, uma linha por mês).

| Coluna | Tipo de Dado | Descrição Detalhada |
|---|---|---|
| `referencia` | Texto (String) | Mês e ano de registro dos dados no formato MM/AAAA. |
| `periodo` | Número Inteiro | Código numérico do mês e ano no formato AAAAMM, usado para facilitar a ordenação cronológica no código. |
| `populacao_estimada` | Número Inteiro | Estimativa da população total do país no período, utilizada como base para o cálculo dos percentuais. |
| `pobreza` | Número Inteiro | Quantidade absoluta estimada de pessoas enquadradas na faixa de pobreza naquele mês. |
| `extrema_pobreza` | Número Inteiro | Quantidade absoluta estimada de pessoas enquadradas na faixa de extrema pobreza naquele mês. |
| `total` | Número Inteiro | Soma consolidada do total de pessoas em situação de pobreza e extrema pobreza no período. |
| `porcentagem_pobreza` | Número Decimal | Proporção percentual da população total que se encontra na faixa de pobreza. |
| `porcentagem_extrema_pobreza` | Número Decimal | Proporção percentual da população total que se encontra na faixa de extrema pobreza. |
| `porcentagem_vulnerabilidade` | Número Decimal | Taxa percentual geral da população em situação de vulnerabilidade social (soma das proporções). |
| `familias_pobreza` | Número Inteiro | Número exato de núcleos familiares registrados na faixa de pobreza. |
| `familias_extrema_pobreza` | Número Inteiro | Número exato de núcleos familiares registrados na faixa de extrema pobreza. |
| `familias_vulnerabilidade` | Número Inteiro | Soma total consolidada de famílias em situação de vulnerabilidade social no mês correspondente. |

---

## 3. Diagnóstico e tratamento inicial dos dados

Antes de partir para os gráficos e para os modelos preditivos, nós carregamos o arquivo no
Python usando o Pandas para checar como os dados estavam estruturados. A tabela tem 122
linhas e 12 colunas, trazendo o registro mês a mês de agosto de 2012 (08/2012) até setembro
de 2022 (09/2022), sem nenhum mês pulado no meio do caminho.

Na hora de conferir a limpeza, vimos que a base já veio bem organizada: não há nenhum valor
nulo nem linha duplicada, então não precisamos apagar registros nem preencher dados
faltantes. Os tipos das colunas também vieram certos para fazer conta, com 8 colunas de
números inteiros e 3 de decimais. Só a coluna `referencia` que veio como texto, mas a gente
vai usar ela só para colocar os nomes nos eixos dos gráficos, já que a coluna `periodo`
resolve a ordem das datas.

### Conferências aplicadas

| Verificação | Resultado |
|---|---|
| Dimensão da tabela | 122 linhas, 12 colunas |
| Valores nulos | 0 |
| Linhas duplicadas | 0 |
| Meses faltando na sequência | 0 |
| `periodo` sem repetição | Confirmado |
| `total` = `pobreza` + `extrema_pobreza` | Confirmado nas 122 linhas |
| `familias_vulnerabilidade` = soma das duas faixas de famílias | Confirmado nas 122 linhas |

As duas últimas conferências são importantes porque garantem que as colunas consolidadas são
coerentes com as colunas de origem, e não um número informado à parte.

### Ajuste nas colunas de porcentagem

As três colunas de porcentagem (`porcentagem_pobreza`, `porcentagem_extrema_pobreza` e
`porcentagem_vulnerabilidade`) estão em formato decimal, variando de 0.04 a 0.33, e precisam
ser multiplicadas por 100 para serem lidas como 4% a 33%.

Durante a construção dos gráficos, porém, percebemos que essas colunas estão arredondadas em
apenas duas casas decimais. Multiplicadas por 100, elas viram valores inteiros, e a curva no
gráfico aparece em degraus, escondendo a variação real de mês para mês. Por isso optamos por
recalcular o percentual a partir dos valores absolutos:

```python
pct_vulnerabilidade = total / populacao_estimada * 100
```

O resultado é equivalente ao da coluna original, porém com a precisão necessária para o
gráfico. As colunas originais foram mantidas intactas na base.

---

## 4. Escopo e variáveis do modelo

Para organizar o que vamos analisar, separamos as variáveis da base em três frentes
principais.

**Recorte de tempo.** Usando `referencia` e `periodo` para acompanhar a evolução ao longo
desses 10 anos e ver em quais momentos os índices subiram ou caíram mais.

**Dados da população.** Com `populacao_estimada`, `pobreza`, `extrema_pobreza`, `total` e as
porcentagens, conseguimos comparar o total de pessoas afetadas, que oscila entre 47,2 milhões
e 64,3 milhões na série, com o crescimento da população do país no mesmo período, que foi de
193,9 milhões para 213,3 milhões de habitantes.

**Domicílios.** Com `familias_pobreza`, `familias_extrema_pobreza` e
`familias_vulnerabilidade`, entendemos o impacto real dentro de casa, mostrando que o total
de famílias em vulnerabilidade variou entre 15,5 milhões e 23,9 milhões ao longo dos meses
analisados.

---

## 5. Análise gráfica

Os gráficos usam **média móvel** como recurso de suavização: cada ponto passa a ser a média
dos meses vizinhos, o que remove a oscilação de curto prazo e deixa a tendência visível. São
usadas duas janelas, de 3 e de 12 meses, sempre junto da linha original em tom mais claro,
para que o dado bruto continue à vista.

O painel traz quatro visões:

1. **Total de pessoas em vulnerabilidade**, com as duas suavizações
2. **Pobreza e extrema pobreza separadas**, que revelam comportamentos distintos
3. **Participação na população estimada**, em percentual
4. **Famílias por faixa**

### O que os gráficos mostram

A leitura do total esconde duas histórias opostas. A faixa de **pobreza** cai de ponta a
ponta da série, saindo de 19,4 milhões em 2012 para 9,5 milhões em 2022. Já a **extrema
pobreza** cai até 2016, fica estável por alguns anos e volta a subir com força a partir de
2021, saindo de 36,8 milhões no ponto mais baixo para 49,1 milhões no fim da série.

O resultado é que o total consolidado desenha um "U": cai de 63,7 milhões em 2012 até o
mínimo de 47,2 milhões em 2019, e sobe de novo para 58,6 milhões em 2022.

Em termos proporcionais, a vulnerabilidade sai de 32,9% da população em 2012, chega ao piso
de 22,5% em 2019 e volta a 27,5% em 2022. Como a população do país cresceu quase 20 milhões
de pessoas no período, a queda proporcional é mais acentuada do que a queda absoluta.

---

## 6. Machine Learning: Random Forest e KNN

### Técnicas escolhidas

Escolhemos dois algoritmos de natureza diferente, para poder comparar:

| Modelo | Como funciona | Parâmetros usados |
|---|---|---|
| **Random Forest** | Combina centenas de árvores de decisão, cada uma treinada com um recorte diferente dos dados, e usa a média das respostas. | `n_estimators=300`, `max_depth=6`, `min_samples_leaf=2` |
| **KNN** | Procura no histórico os meses mais parecidos com o atual e responde com a média do que ocorreu naqueles meses. | `n_neighbors=5`, com `StandardScaler` |

### Normalização

O KNN decide por distância entre pontos. Como as colunas têm escalas muito diferentes, sem
normalizar a distância seria dominada pelas colunas de maior magnitude. Por isso o modelo foi
montado dentro de um `Pipeline` com `StandardScaler`, que coloca todas as variáveis na mesma
escala. O uso do `Pipeline` também garante que a normalização seja aprendida apenas com os
dados de treino.

### Modelagem da variável alvo

Em vez de prever diretamente o **nível** (quantas pessoas haverá em determinado mês), os
modelos preveem a **variação mensal**, ou seja, quanto o total muda de um mês para o outro. A
reconstrução do nível é feita somando a variação prevista ao último valor conhecido.

Essa decisão foi tomada por uma limitação conhecida dos modelos baseados em árvore: o Random
Forest responde pela média dos exemplos de cada folha, e média de valores conhecidos nunca
ultrapassa o maior valor conhecido. Treinado diretamente no nível, o modelo ficaria preso ao
teto do período de treino. Ao prever a variação, o alvo fica centrado próximo de zero, dentro
da faixa que o modelo conhece.

As variáveis preditoras são as variações anteriores (1, 2, 3 e 12 meses atrás), as médias das
variações dos últimos 3 e 12 meses, e o mês do ano. Todas são calculadas com `shift`, de modo
que nenhuma informação do mês previsto entre no cálculo.

### Divisão treino e teste

Por se tratar de uma série ordenada no tempo, a divisão **não** foi feita por sorteio
aleatório. Sortear as linhas permitiria ao modelo treinar com meses posteriores aos que
precisa prever, o que produziria uma avaliação otimista e irreal. A divisão respeita a
cronologia:

- **Treino:** 85 meses, de 09/2013 a 09/2020
- **Teste:** 24 meses, de 10/2020 a 09/2022

---

## 7. Validação dos modelos

Para validar, treinamos os modelos sem os últimos 24 meses e pedimos que previssem justamente
esse período. Como já sabemos o que aconteceu, dá para colocar o previsto e o real lado a
lado e verificar se os valores ficaram próximos.

### Previsto x Real (amostra do período de teste)

| Mês | Real | Random Forest | KNN |
|---|---|---|---|
| 10/2020 | 48.839.108 | 47.178.917 | 47.688.461 |
| 11/2020 | 47.895.441 | 47.562.489 | 48.477.169 |
| 12/2020 | 48.079.296 | 47.560.454 | 47.434.597 |
| 01/2021 | 48.279.265 | 48.284.987 | 46.892.772 |
| ... | ... | ... | ... |
| 06/2022 | 56.311.977 | 54.754.532 | 55.963.970 |
| 07/2022 | 56.869.723 | 55.296.605 | 56.358.986 |
| 08/2022 | 57.716.117 | 55.778.147 | 56.916.732 |
| 09/2022 | 58.565.884 | 56.537.746 | 57.415.586 |

A tabela completa dos 24 meses, com a coluna de diferença destacada por cor, está no painel
em Streamlit.

### Resumo do erro

| Modelo | Erro médio (pessoas) | Erro médio (%) | Maior erro isolado |
|---|---|---|---|
| **KNN** | 953.655 | 1,84% | 4.017.744 |
| **Random Forest** | 1.345.102 | 2,54% | 3.660.009 |

### Leitura dos resultados

Os dois modelos acompanham o movimento real da série de perto, com erro médio abaixo de 3%.
O **KNN teve o menor erro médio**, ficando cerca de 390 mil pessoas mais próximo do real do
que o Random Forest. Em compensação, o Random Forest apresentou o **menor erro máximo**, ou
seja, é mais consistente nos meses atípicos, enquanto o KNN acerta mais na média mas erra
mais feio nos picos.

Vale registrar uma observação metodológica: os dois modelos ficam ligeiramente **abaixo** do
real durante quase todo o período de teste. Isso ocorre porque o período de treino (até
setembro de 2020) é majoritariamente de queda, enquanto o período de teste é de alta
acentuada. O modelo aprendeu um comportamento de queda e precisou prever uma subida. Como as
causas dessa inversão (pandemia, auxílio emergencial e a substituição do Bolsa Família pelo
Auxílio Brasil) são de natureza externa e não estão entre as colunas do arquivo, nenhum
modelo treinado apenas com o histórico da própria série teria como antecipá-las.

---

## 8. Projeção para os próximos anos

Validados os modelos, ambos foram retreinados com a série completa e usados para projetar 60
meses à frente, até setembro de 2027. A projeção é recursiva: a previsão de um mês passa a
ser a base da previsão do mês seguinte.

Foi aplicado um fator de amortecimento de 0,9 sobre a variação prevista a cada passo. Sem
ele, o modelo replica indefinidamente a última tendência observada e chega a valores
inconsistentes com a realidade demográfica do país. O amortecimento faz a tendência perder
força gradualmente, prática usual em projeções de horizonte longo.

| Setembro de | Random Forest | KNN |
|---|---|---|
| 2023 | 62.559.538 | 62.237.956 |
| 2024 | 62.046.147 | 61.558.276 |
| 2025 | 61.726.863 | 61.248.711 |
| 2026 | 61.611.229 | 61.150.291 |
| 2027 | 61.577.408 | 61.122.573 |

Os dois modelos chegam a resultados próximos entre si, com diferença abaixo de 1%: uma alta
nos primeiros 12 meses, em continuidade ao movimento de 2021 e 2022, seguida de estabilização
em torno de 61 a 62 milhões de pessoas.

---

## 9. Conclusões

**A série tem duas histórias sobrepostas.** A pobreza caiu de forma contínua durante toda a
década analisada, enquanto a extrema pobreza inverteu a trajetória a partir de 2021. Olhar
apenas o total consolidado esconde esse movimento, o que reforça a importância de desagregar
as faixas na análise.

**A queda proporcional foi maior que a absoluta.** Como a população cresceu de 193,9 para
213,3 milhões no período, a participação da vulnerabilidade saiu de 32,9% para 27,5%, mesmo
com o total absoluto voltando a subir no fim da série.

**Os modelos validaram com erro médio abaixo de 3%.** O KNN obteve o menor erro médio (1,84%)
e o Random Forest a maior consistência nos meses atípicos. Para uso prático, a escolha entre
os dois depende de qual característica importa mais: proximidade média ou previsibilidade do
pior caso.

**O limite da previsão é a própria base.** Os erros se concentram exatamente no período de
inversão de tendência, e a causa dessa inversão é externa aos dados. Para uma projeção mais
confiável seria necessário incorporar variáveis econômicas (desemprego, inflação, salário
mínimo) e informações sobre o valor e o alcance dos programas de transferência de renda. A
projeção apresentada deve ser lida como um cenário de continuidade, não como garantia.

---

## 10. Estrutura do repositório

```
.
├── main.py           # painel em Streamlit: gráficos, validação e projeção
├── indices.csv       # base de dados utilizada
├── pyproject.toml    # dependências do projeto (gerenciadas com uv)
├── uv.lock           # versões travadas das dependências
├── README.md         # este documento
└── src/reinaldo/     # pacote do projeto
```

---

## 11. Como executar

O projeto usa [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente e as dependências.

```bash
uv sync
```

```bash
uv run streamlit run main.py
```

O painel abre no navegador em `http://localhost:8501`.

**Dependências principais:** Python 3.14, pandas, numpy, matplotlib, scikit-learn e
streamlit. As versões exatas estão travadas em `uv.lock`.
