import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ===============================
# CONFIGURAÇÃO DA PÁGINA
# ===============================
st.set_page_config(page_title="IA Logística", layout="wide")
st.title("📦 Portal IA – Logística")

# ===============================
# CARREGAR PLANILHA
# ===============================
@st.cache_data
def carregar_dados():
    df_estoque = pd.read_excel("ESTOQUE.xlsx", sheet_name="ESTOQUE")
    df_carteira = pd.read_excel("ESTOQUE.xlsx", sheet_name="CARTEIRA VS ESTOQUE")
    df_entrega = pd.read_excel("ESTOQUE.xlsx", sheet_name="ACOMPANHAMENTO DE ENTREGA")
    return df_estoque, df_carteira, df_entrega

df_estoque, df_carteira, df_entrega = carregar_dados()

# ===============================
# BASE TEXTUAL PARA IA
# ===============================
def criar_textos():
    textos = []

    for _, r in df_estoque.iterrows():
        textos.append(
            f"Produto {r['DESCRICAO']} possui {r['QTDE']} unidades em estoque."
        )

    for _, r in df_carteira.iterrows():
        textos.append(
            f"O produto {r['DESCRICAO']} tem {r['QTDE_CARTEIRA']} em carteira "
            f"e {r['TOTAL_GERAL_ESTOQUE']} em estoque."
        )

    for _, r in df_entrega.iterrows():
        textos.append(
            f"A nota fiscal {r['NF']} do cliente {r['CLIENTE']} está com status {r['STATUS']}."
        )

    return textos

base_textos = criar_textos()

# ===============================
# EMBEDDINGS + FAISS
# ===============================
@st.cache_resource
def criar_indice(textos):
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    emb = modelo.encode(textos)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return modelo, index

modelo_emb, indice = criar_indice(base_textos)

# ===============================
# MODELO DE IA (GRATUITO)
# ===============================
@st.cache_resource
def carregar_ia():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=200
    )

llm = carregar_ia()

# ===============================
# INTERFACE
# ===============================
pergunta = st.text_input("❓ Pergunte algo sobre estoque, carteira ou entregas:")

if pergunta:
    vetor = modelo_emb.encode([pergunta])
    _, ids = indice.search(np.array(vetor), k=5)

    contexto = " ".join([base_textos[i] for i in ids[0]])

    prompt = f"""
    Use SOMENTE os dados abaixo para responder:

    {contexto}

    Pergunta: {pergunta}
    Resposta:
    """

    resposta = llm(prompt)[0]["generated_text"]
    st.success(resposta)
