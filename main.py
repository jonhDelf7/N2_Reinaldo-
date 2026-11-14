import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp 


# configuração da página
st.set_page_config(
    page_title= "Análise Estatística e Probabilisca de Dados",
    layout= "wide"
)
st.markdown("# Estudo de caso: Índice de pobreza ")


#Escolha e tratamento de dados
db = pd.read_csv("indices.csv")
db.drop(columns=["populacao_estimada","porcentagem_pobreza","porcentagem_extrema_pobreza","porcentagem_vulnerabilidade","familias_pobreza","familias_extrema_pobreza","familias_vulnerabilidade"]
)

st.dataframe(db.style.highlight_max(axis=0))