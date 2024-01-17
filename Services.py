import streamlit as st




st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

with open('./style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.write("# Welcome to PromptlyTech Enterprise Grade RAG Systems 👋")

st.markdown(
    """
    Services We provide:
    - Automatic Prompt Generation
    - Automatic Evaluation Data Generation
    - Prompt Testing and Ranking Service:
"""
)
