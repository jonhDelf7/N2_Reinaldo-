"""
ATIVIDADE PRÁTICA: ANÁLISE DE DADOS E MACHINE LEARNING
Estudo de caso: índice de pobreza no Brasil (08/2012 a 09/2022)

Integrantes:
    Gustavo Samuel da Cruz Morais
    Abner Luiz Pascoal de Oliveira
    João Pedro Delfino Teixeira
    João Gabriel Arantes Sena

Para rodar:  uv run streamlit run main.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# configuração da página
st.set_page_config(
    page_title="Análise Estatística e Probabilística de Dados",
    layout="wide",
)

st.markdown("# Estudo de caso: Índice de pobreza")
st.caption(
    "Comportamento da pobreza e da extrema pobreza no Brasil entre agosto de 2012 "
    "e setembro de 2022, com projeção por Random Forest e KNN."
)

AZUL = "#1f6feb"
VERMELHO = "#d1242f"
ROXO = "#6f42c1"
VERDE = "#1a7f37"
GRAFITE = "#24292f"

PISTAS = ["mudanca_1", "mudanca_2", "mudanca_3", "mudanca_12", "media_3", "media_12", "mes"]
MESES_PREVISTOS = 60  # 5 anos
FREIO = 0.9  # segura a tendência para a projeção não crescer sem limite


# Escolha e tratamento de dados
@st.cache_data
def carregar_dados():
    """Lê o CSV, coloca a data no índice e cria as colunas de percentual."""
    db = pd.read_csv("indices.csv")
    db["data"] = pd.to_datetime(db["referencia"], format="%m/%Y")
    db = db.sort_values("data").set_index("data")

    # As colunas de porcentagem do arquivo vêm em formato decimal (0.33 = 33%), mas
    # arredondadas em duas casas. Multiplicadas por 100 viram degraus de número inteiro
    # e a curva perde o detalhe. Por isso recalculamos a partir dos valores absolutos,
    # que dá o mesmo percentual com precisão. As colunas originais ficam intactas.
    db["pct_pobreza"] = db["pobreza"] / db["populacao_estimada"] * 100
    db["pct_extrema_pobreza"] = db["extrema_pobreza"] / db["populacao_estimada"] * 100
    db["pct_vulnerabilidade"] = db["total"] / db["populacao_estimada"] * 100
    return db


@st.cache_data
def preparar_para_modelo(db):
    """Monta as colunas que os modelos usam: as mudanças de um mês para o outro."""
    dados = db.copy()
    dados["mudanca"] = dados["total"].diff()

    for n in [1, 2, 3, 12]:
        dados[f"mudanca_{n}"] = dados["mudanca"].shift(n)
    dados["media_3"] = dados["mudanca"].shift(1).rolling(3).mean()
    dados["media_12"] = dados["mudanca"].shift(1).rolling(12).mean()
    dados["mes"] = dados.index.month
    dados["total_anterior"] = dados["total"].shift(1)

    return dados.dropna(subset=PISTAS + ["mudanca", "total_anterior"])


def criar_modelos():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    }


@st.cache_data
def validar(dados, meses_teste=24):
    """Treina sem os últimos meses e compara o previsto com o que realmente aconteceu."""
    treino = dados.iloc[:-meses_teste]
    teste = dados.iloc[-meses_teste:]

    comparacao = pd.DataFrame({"Real": teste["total"]})

    for nome, modelo in criar_modelos().items():
        modelo.fit(treino[PISTAS], treino["mudanca"])
        previsto = teste["total_anterior"].to_numpy() + modelo.predict(teste[PISTAS])
        comparacao[nome] = previsto
        comparacao[f"Diferença {nome}"] = previsto - teste["total"].to_numpy()

    comparacao.index = comparacao.index.strftime("%m/%Y")
    return comparacao


@st.cache_data
def projetar(dados, meses=MESES_PREVISTOS):
    """Prevê mês a mês, usando a própria previsão como base para o mês seguinte."""
    resultado = {}

    for nome, modelo in criar_modelos().items():
        modelo.fit(dados[PISTAS], dados["mudanca"])

        totais = dados["total"].astype(float).copy()
        mudancas = dados["mudanca"].astype(float).copy()
        datas, valores = [], []

        for passo in range(meses):
            proximo = totais.index[-1] + pd.DateOffset(months=1)
            pistas = {
                "mudanca_1": mudancas.iloc[-1],
                "mudanca_2": mudancas.iloc[-2],
                "mudanca_3": mudancas.iloc[-3],
                "mudanca_12": mudancas.iloc[-12],
                "media_3": mudancas.iloc[-3:].mean(),
                "media_12": mudancas.iloc[-12:].mean(),
                "mes": proximo.month,
            }
            mudanca = float(modelo.predict(pd.DataFrame([pistas])[PISTAS])[0])
            mudanca *= FREIO ** (passo + 1)
            novo = float(totais.iloc[-1]) + mudanca

            totais.loc[proximo] = novo
            mudancas.loc[proximo] = mudanca
            datas.append(proximo)
            valores.append(novo)

        resultado[nome] = pd.Series(valores, index=pd.DatetimeIndex(datas))

    return resultado


# Gráficos
plt.rcParams.update({
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
})


def novo_grafico():
    figura, eixo = plt.subplots(figsize=(11, 4.5))
    eixo.grid(alpha=0.25)
    eixo.spines[["top", "right"]].set_visible(False)
    return figura, eixo


def eixo_em_milhoes(eixo):
    eixo.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v / 1_000_000:.0f} mi")
    )


def grafico_total(db):
    serie = db["total"]
    figura, eixo = novo_grafico()
    eixo.plot(serie.index, serie, color=AZUL, alpha=0.30, label="Cada mês")
    eixo.plot(serie.index, serie.rolling(3, center=True, min_periods=1).mean(),
              color=AZUL, linewidth=2, label="Média de 3 meses")
    eixo.plot(serie.index, serie.rolling(12, center=True, min_periods=1).mean(),
              color=VERMELHO, linewidth=3, label="Média de 12 meses")
    eixo.set_title("Pessoas em pobreza ou extrema pobreza")
    eixo_em_milhoes(eixo)
    eixo.legend()
    figura.tight_layout()
    return figura


def grafico_faixas(db):
    figura, eixo = novo_grafico()
    for coluna, nome, cor in [
        ("extrema_pobreza", "Extrema pobreza", VERMELHO),
        ("pobreza", "Pobreza", AZUL),
    ]:
        serie = db[coluna]
        eixo.plot(serie.index, serie, color=cor, alpha=0.25)
        eixo.plot(serie.index, serie.rolling(12, center=True, min_periods=1).mean(),
                  color=cor, linewidth=3, label=nome)
    eixo.set_title("Pobreza e extrema pobreza, separadas")
    eixo_em_milhoes(eixo)
    eixo.legend()
    figura.tight_layout()
    return figura


def grafico_percentual(db):
    figura, eixo = novo_grafico()
    for coluna, nome, cor in [
        ("pct_vulnerabilidade", "Vulnerabilidade (total)", ROXO),
        ("pct_extrema_pobreza", "Extrema pobreza", VERMELHO),
        ("pct_pobreza", "Pobreza", AZUL),
    ]:
        eixo.plot(db.index, db[coluna], color=cor, linewidth=2.2, label=nome)
    eixo.set_title("Participação na população estimada")
    eixo.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    eixo.legend()
    figura.tight_layout()
    return figura


def grafico_familias(db):
    figura, eixo = novo_grafico()
    for coluna, nome, cor in [
        ("familias_vulnerabilidade", "Vulnerabilidade (total)", ROXO),
        ("familias_extrema_pobreza", "Extrema pobreza", VERMELHO),
        ("familias_pobreza", "Pobreza", AZUL),
    ]:
        serie = db[coluna]
        eixo.plot(serie.index, serie.rolling(12, center=True, min_periods=1).mean(),
                  color=cor, linewidth=2.5, label=nome)
    eixo.set_title("Famílias por faixa")
    eixo_em_milhoes(eixo)
    eixo.legend()
    figura.tight_layout()
    return figura


def grafico_validacao(comparacao):
    figura, eixo = novo_grafico()
    datas = pd.to_datetime(comparacao.index, format="%m/%Y")
    eixo.plot(datas, comparacao["Real"], color=GRAFITE, linewidth=3.5, label="Real")
    eixo.plot(datas, comparacao["Random Forest"], color=VERDE, linewidth=2,
              linestyle="--", label="Random Forest")
    eixo.plot(datas, comparacao["KNN"], color=VERMELHO, linewidth=2,
              linestyle="--", label="KNN")
    eixo.set_title("Validação: previsto x real nos últimos 24 meses")
    eixo_em_milhoes(eixo)
    eixo.legend()
    figura.tight_layout()
    return figura


def grafico_projecao(db, previsoes):
    figura, eixo = novo_grafico()
    eixo.plot(db.index, db["total"], color=GRAFITE, linewidth=2.5, label="Observado")

    ultimo_mes = db.index[-1]
    ultimo_valor = db["total"].iloc[-1]

    for nome, cor in [("Random Forest", VERDE), ("KNN", VERMELHO)]:
        serie = previsoes[nome]
        eixo.plot([ultimo_mes] + list(serie.index),
                  [ultimo_valor] + list(serie.values),
                  color=cor, linewidth=2.5, linestyle="--", label=f"Projeção: {nome}")

    eixo.axvline(ultimo_mes, color=GRAFITE, linestyle=":", linewidth=1.5)
    eixo.set_title("Projeção até setembro de 2027")
    eixo_em_milhoes(eixo)
    eixo.legend(loc="upper left")
    figura.tight_layout()
    return figura


# Montagem da página
db = carregar_dados()
dados = preparar_para_modelo(db)
comparacao = validar(dados)
previsoes = projetar(dados)

coluna1, coluna2, coluna3, coluna4 = st.columns(4)
coluna1.metric("Meses analisados", len(db))
coluna2.metric("Pessoas em 09/2022", f"{db['total'].iloc[-1] / 1_000_000:.1f} mi")
coluna3.metric(
    "Famílias em 09/2022", f"{db['familias_vulnerabilidade'].iloc[-1] / 1_000_000:.1f} mi"
)
coluna4.metric("Da população", f"{db['pct_vulnerabilidade'].iloc[-1]:.0f}%")

st.divider()

st.markdown("## Panorama da série")
esquerda, direita = st.columns(2)
esquerda.pyplot(grafico_total(db))
direita.pyplot(grafico_faixas(db))

esquerda, direita = st.columns(2)
esquerda.pyplot(grafico_percentual(db))
direita.pyplot(grafico_familias(db))

st.divider()

st.markdown("## Validação dos modelos")
st.write(
    "Os modelos foram treinados sem os últimos 24 meses e depois tiveram que prever "
    "justamente esse período. Como já sabemos o que aconteceu, a tabela abaixo mostra "
    "lado a lado o valor previsto e o valor real de cada mês."
)

st.pyplot(grafico_validacao(comparacao))

st.dataframe(
    comparacao.style.format("{:,.0f}").background_gradient(
        axis=0, subset=["Diferença Random Forest", "Diferença KNN"], cmap="RdYlGn_r"
    ),
    width="stretch",
)

st.divider()

st.markdown("## Projeção para os próximos 5 anos")
st.pyplot(grafico_projecao(db, previsoes))

tabela_projecao = pd.DataFrame(
    {nome: serie[serie.index.month == 9] for nome, serie in previsoes.items()}
)
tabela_projecao.index = tabela_projecao.index.strftime("%m/%Y")
st.dataframe(tabela_projecao.style.format("{:,.0f}"), width="stretch")

st.caption("A projeção é um cenário de inércia, não uma garantia.")

st.divider()

st.markdown("## Base de dados")
st.dataframe(db.style.highlight_max(axis=0), width="stretch")
