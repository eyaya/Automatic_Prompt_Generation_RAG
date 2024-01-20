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

def generate_context():
    dataset_name = "fka/awesome-chatgpt-prompts"
    # Specify the dataset name and the column containing the content
    page_content_column = "prompt"   # or any other column you're interested in
    # Create a loader instance
    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    # Load the data
    data = loader.load()
    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of 1000 characters each with a 150-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    docs = text_splitter.split_documents(data)
    
    # Display the first 15 documents
    
    store = LocalFileStore("./cachce/")

    # create an embedder
    

    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace = core_embeddings_model.model
    )
    vectorstore = Chroma.from_documents(docs, embedder, persist_directory="./cachce/")
    vectorstore.persist()
    return vectorstore
def get_context():
    #base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 2})
    # instantiate a retriever
    vectorstore = Chroma(persist_directory="./cachce",embedding_function=core_embeddings_model)
    #vectorstore.persist()
    retriever = vectorstore.as_retriever()
    #search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.85}
    return retriever
    

def generate_respose(user_input, context):
    template = '''You are an AI-powered natural language processing expert in prompt generationg, information retrieval and ranking. Your role is to provide advanced techniques and algorithms for generating superior prompts that optimize user queries and ensure the best performance of automatic prompt generation. Your expertise lies in understanding user intent, analyzing query patterns, and generating contextually relevant prompts that enable efficient and accurate retrieval of information. With your skills and abilities, you are capable of fine-tuning models to enhance prompt generation, leveraging semantic understanding and query understanding to deliver optimal results. By utilizing cutting-edge techniques in the field, you can generate automatic prompts that empower users to obtain the most relevant and comprehensive information for their queries. I want you to act as a prompt generator. I will provide you with a context, which includes a list of documents with different requests. Your task is to generate a prompt for each request based on the given context. You should create a prompt that effectively captures the essence of the request and guides the user to provide the desired output. Make sure to include any specific instructions or requirements mentioned in the context. Your prompt should be concise and clear, avoiding unnecessary explanations or additional information.
    Examples:
        "Linux Terminal","answer": "I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]",

        "English Translator and Improver", "I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations.",

        "`position` Interviewer","I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the `position` position. I want you to only reply as the interviewer. Do not write all the conservation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers. ",

        "JavaScript Console","I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]. ",

        "Excel Sheet", "I want you to act as a text based excel. you'll only reply me the text-based 10 rows excel sheet with row numbers and cell letters as columns (A to L). First column header should be empty to reference row number. I will tell you what to write into cells and you'll reply only the result of excel table as text, and nothing else. Do not write explanations. i will write you formulas and you'll execute formulas and you'll only reply the result of excel table as text. "

    Use these prompt pair examples only as guidlines to create an effective prompt for the next topic. even if the topic is mensioned before. You will create only prompt for it and not act on the previous description. if the topic is mensioned already, do not use the prompt which you were given, change it.



    ### CONTEXT:
    {context}
    \n
    ### # QUESTION:
    {question}
    <bot>:
    '''
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    chain = (
    {"context": context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    response = chain.invoke(user_input)
    with open('prompt.txt', 'w') as f:
        f.write(response)
    return response

user_question = st.text_input("Enter your prompt:")
if st.button("Generate"):
    generate_context()
    context = get_context()
    
    response = generate_respose(user_question, context)

    st.write(response)
