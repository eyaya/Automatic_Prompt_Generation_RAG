import streamlit as st
# Import necessary modules
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from openai import OpenAI
from dotenv import load_dotenv

import os
load_dotenv()
OPENAI_API_KEY = os.environ.get('openai_api_key')

# Loading Data from huggingface for the directive of the prompt.
dataset_name = "fka/awesome-chatgpt-prompts"
# Specify the dataset name and the column containing the content
page_content_column = "prompt"   # or any other column you're interested in
# Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
# Load the data
data = loader.load()

# Page configuration
st.set_page_config(page_title="Automatic Prompt Generator", page_icon="📈")



# Rest of the page
st.markdown("# Automatic Prompt Generator")
#st.sidebar.header("Automatic Prompt Generator")

st.text_area('Ask for a prompt', value="", label_visibility="visible")

input = st.empty()
txt = input.text_input("Insert text:")
bt = st.button("Text01")

if bt:
    txt = "Text01"
    input.text_input("Insert text:", value=txt)

st.write(txt)