import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not configured.")
    st.stop()

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your documents here and start asking questions", type="pdf")  # ["pdf", "docx", "txt"]

#Extract content from PDF and chunk it
if file is not None:
    #extract text from it
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    # st.write(text)
   
   # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n",".", " ", ""],
        chunk_size=500,
        chunk_overlap=50
    )

    text_chunks = text_splitter.split_text(text)
    # st.write(text_chunks)

    #Generating embeddings and storing them in a vector database

    #Generate
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     openai_api_key=OPENAI_API_KEY
    # )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #store
    vector_store = FAISS.from_texts(text_chunks, embeddings)

    #get user query and retrieve relevant documents
    user_question = st.text_input("Ask a question about your document:")

    #generate answers
    #question -> embedding -> SimilaritySearch -> results to LLM -> response (CHAIN)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
        )

    #define the LLM and prompts
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0.3,
    #     max_tokens=1000,
    #     openai_api_key=OPENAI_API_KEY
    # )
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.3
    )


    #provide the prompts
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])


    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)
