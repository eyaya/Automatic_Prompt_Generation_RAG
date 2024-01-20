import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from utils import read_from_file, delete_file
load_dotenv()

OPENAI_API_KEY = os.environ.get('openai_api_key')
store = LocalFileStore("./cachce/")
# Define the path for generated embeddings
DB_FAISS_PATH = './db_faiss/db_faiss'

def create_chat(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=30,
        length_function=len
    )
    chunks = text_splitter.split_text(text)


    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #vectorstore = Chroma.from_texts(text_chunks, embeddings, persist_directory="./cachce/")
    vectorstore =FAISS.from_texts(texts=chunks, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
        


    llm = ChatOpenAI(model_name="gpt-3.5-turbo",verbose=True, temperature=0)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    read_prompt = read_from_file('prompt.txt')

    system_template = read_prompt+"""
    you only answer the questions related to the given context bellow. If there is no information you need to replay 'I have not .found the answer to your question'
    ----
    ### CONTEXT:
    {context}
    ----
    """
    print(system_template)
    user_template = "Question:```{question}```"
    messages = [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )
    print(system_template)
    with open('prompt.txt', 'w') as f:
        f.write(system_template)
    #prompt = ChatPromptTemplate.from_template(template)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type='stuff',
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return conversation_chain
def main():
    load_dotenv()
    st.set_page_config(page_title="PolyTech", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PolyTech Automatic Prompt Generation Service :books:")
    
    

    
    st.sidebar.subheader("Your Documents")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    
    if pdf_docs:
        
        conversation_chain = create_chat(pdf_docs)
        def handle_userinput(user_question):
            result = conversation_chain({"question": user_question, "chat_history": st.session_state['history']})
            st.session_state['history'].append((conversation_chain, result["answer"]))
            return result["answer"]

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me about your documents 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Ask me about your data 👉 (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = handle_userinput(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history 
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


if __name__ == '__main__':
    main()