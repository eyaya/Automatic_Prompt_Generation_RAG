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
def generate_directive(user_input):
    # Loading Data from huggingface for the directive of the prompt.
    dataset_name = "fka/awesome-chatgpt-prompts"
    # Specify the dataset name and the column containing the content
    page_content_column = "prompt"   # or any other column you're interested in
    # Create a loader instance
    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    # Load the data
    data = loader.load()

    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of 1000 characters each with a 150-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    docs = text_splitter.split_documents(data)

    store = LocalFileStore("./cachce/")

    # create an embedder
    core_embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace = core_embeddings_model.model
    )


    vectorstore = Chroma.from_documents(docs, embedder, persist_directory="./cachce/")

    #base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 2})
    # instantiate a retriever
    retriever = vectorstore.as_retriever()

    # Directive template
    examples = [
        {
            "query": "Linux Terminal",
            "answer": "I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]",

        }, 
        {
            "query": "English Translator and Improver",
            "answer": "I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations.",
        }, 
        {
            "query": "`position` Interviewer",
            "answer": "I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the `position` position. I want you to only reply as the interviewer. Do not write all the conservation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers. ",
        }, 
        {
            "query": "JavaScript Console",
            "answer": "I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]. ",
        }, 
        {
            "query": "Excel Sheet",
            "answer": "I want you to act as a text based excel. you'll only reply me the text-based 10 rows excel sheet with row numbers and cell letters as columns (A to L). First column header should be empty to reference row number. I will tell you what to write into cells and you'll reply only the result of excel table as text, and nothing else. Do not write explanations. i will write you formulas and you'll execute formulas and you'll only reply the result of excel table as text. ",
        }
    ]

    template = '''"Break down the prompt genetation step by step based on the following prompt pairs = "Linux Terminal","I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]"
    "English Translator and Improver","I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations."
    "`position` Interviewer","I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the `position` position. I want you to only reply as the interviewer. Do not write all the conservation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers. "
    "JavaScript Console","I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets [like this]. "
    "Excel Sheet","I want you to act as a text based excel. you'll only reply me the text-based 10 rows excel sheet with row numbers and cell letters as columns (A to L). First column header should be empty to reference row number. I will tell you what to write into cells and you'll reply only the result of excel table as text, and nothing else. Do not write explanations. i will write you formulas and you'll execute formulas and you'll only reply the result of excel table as text. "
    "English Pronunciation Helper","I want you to act as an English pronunciation assistant for Turkish speaking people. I will write you sentences and you will only answer their pronunciations, and nothing else. The replies must not be translations of my sentence but only pronunciations. Pronunciations should use Turkish Latin letters for phonetics. Do not write explanations on replies."
    "Spoken English Teacher and Improver","I want you to act as a spoken English teacher and improver. I will speak to you in English and you will reply to me in English to practice my spoken English. I want you to keep your reply neat, limiting the reply to 100 words. I want you to strictly correct my grammar mistakes, typos, and factual errors. I want you to ask me a question in your reply. Now let's start practicing, you could ask me a question first. Remember, I want you to strictly correct my grammar mistakes, typos, and factual errors."
    "Travel Guide","I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. "
    "Plagiarism Checker","I want you to act as a plagiarism checker. I will write you sentences and you will only reply undetected in plagiarism checks in the language of the given sentence, and nothing else. Do not write explanations on replies. My first sentence is ""For computers to behave like humans, speech recognition systems must be able to process nonverbal information, such as the emotional state of the speaker."""
    "Character from Movie/Book/Anything","I want you to act like [character] from [series]. I want you to respond and answer like [character] using the tone, manner and vocabulary [character] would use. Do not write any explanations. Only answer like [character]. You must know all of the knowledge of [character]. "
    "Advertiser","I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. "
    use these topics, prompt pair examples only as guidlines to create an effective prompt for the next topic. even if the topic is mensioned before. You will create
    only prompt for it and not act on the previous description. if the topic is mensioned already,
    do not use the prompt which you were given, change it.
    ### EXAMPLES
    {examples}
    ### CONTEXT
    {context}
    ### QUESTION
    Question: {question}
    \n
    <bot>:
    '''
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    response = chain.invoke(user_input)

    return response


# Page configuration
st.set_page_config(page_title="Automatic Prompt Generator", page_icon="📈")



# Rest of the page
st.markdown("# Automatic Prompt Generator")
#st.sidebar.header("Automatic Prompt Generator")

query = st.text_area('Ask for a prompt', value="", label_visibility="visible")

out_put = generate_directive(query)

bt = st.button("Submit")

if bt:
    input = st.empty()
    txt = input.text_input(f"{out_put}")

st.write(txt)