import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Notebook", layout="wide")
st.title("📚 Notebook")

llm = ChatOpenAI(
    model="gpt-oss:120b",
    base_url="https://ollama.com/v1",
    api_key=os.getenv("OLLAMA_API_KEY")
)

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "summary" not in st.session_state:
    st.session_state.summary = None

if uploaded_files and st.session_state.vectorstore is None:

    text = ""
    for f in uploaded_files:
        text += read_pdf(f) + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    st.session_state.vectorstore = vectorstore

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    summary_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant.

Summarize this document in a clear, structured way:

{context}
""")

    docs = retriever.invoke("summarize this document")
    context = "\n\n".join([d.page_content for d in docs])

    summary_chain = summary_prompt | llm | StrOutputParser()

    st.session_state.summary = summary_chain.invoke({"context": context})

if st.session_state.summary:
    st.markdown("## Document Summary")
    st.write(st.session_state.summary)

query = st.text_input("Ask a question")
ques=st.button("Get Answer")

if ques and st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    qa_prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context below:

Context:
{context}

Question:
{question}
""")

    qa_chain = qa_prompt | llm | StrOutputParser()

    result = qa_chain.invoke({
        "context": context,
        "question": query
    })

    st.markdown("## 💬 Answer")
    st.write(result)