# Import necessary modules
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from tqdm.auto import tqdm
from langchain import text_splitter
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import List
from langchain.schema import Document
from uuid import uuid4

from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st
#
# import streamlit as st
OPENAI_API_KEY = os.environ.get('openai_api_key')

st.subheader("Your APG!")

core_embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    

def generate_respose(user_input, context):
    template = '''You are an AI-powered natural language processing expert in prompt generationg, information retrieval and ranking. Your role is to provide {n_tests} advanced prompts that optimize user queries and ensure the best performance of automatic prompt generation. 
    the context provided.
    the output must be in a json format.

    example:
    [
        {
            "question": "Linux Terminal",
            "prompt": "I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]".
        },
        {
            "question": "English Translator and Improver",
            "prompt": "I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations.",
        },
        {
            "question": "`position` Interviewer",
            "prompt": "I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the `position` position. I want you to only reply as the interviewer. Do not write all the conservation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait

        },
        {
            "question": "JavaScript Console",
            "prompt": "I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]."
        }
    ]    
     
    Even if the topic is mensioned before. You will create only prompt for it and not act on the previous description. if the topic is mensioned already, do not use the prompt which you were given, change it.

    contex: {context}
    \n
    question:
    {question}    
    <bot>:
    '''
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    chain = (
    {"context": context,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    response = chain.invoke(user_input)
    with open('prompt.txt', 'w') as f:
        f.write(response)
    return response

user_question = st.text_input("How man Prompts would you like to generate")
if st.button("Generate"):
    vectorstore = Chroma(persist_directory="./cachce",embedding_function=core_embeddings_model)
    vectordb = Chroma(persist_directory="./db_faiss", embedding_function=core_embeddings_model)

    context = vectordb.as_retriever()

    response = generate_respose(user_question, context)

    st.write(response)
