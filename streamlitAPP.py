import os
from pathlib import Path
import streamlit as st

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(
    page_title="NVIDIA NIM Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Q&A using NVIDIA NIM")
st.write(
    "Ask questions about PDF documents using NVIDIA NIM, "
    "LangChain, FAISS, and Retrieval-Augmented Generation."
)

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)


# -----------------------------
# API KEY CHECK
# -----------------------------

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error(
        "NVIDIA_API_KEY is missing. "
        "Add it to your local .env file before running the application."
    )
    st.stop()


# -----------------------------
# INITIALIZE LLM
# -----------------------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key
)


# -----------------------------
# VECTOR EMBEDDING FUNCTION
# -----------------------------

def create_vector_store(uploaded_files):

    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    loader = PyPDFDirectoryLoader(str(temp_dir))
    documents = loader.load()

    if not documents:
        st.warning("No readable PDF content was found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    document_chunks = text_splitter.split_documents(documents)

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b"
    )

    st.session_state.vectors = FAISS.from_documents(
        document_chunks,
        embeddings
    )

    st.session_state.document_count = len(uploaded_files)
    st.session_state.chunk_count = len(document_chunks)


# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------

if st.button("Create Document Embeddings"):

    with st.spinner("Reading PDFs and creating embeddings..."):
        create_vector_store(uploaded_files)

    if "vectors" in st.session_state:

        st.success(
            f"Embeddings created successfully from "
            f"{st.session_state.document_count} pages "
            f"and {st.session_state.chunk_count} chunks."
        )


# -----------------------------
# USER QUESTION
# -----------------------------

question = st.text_input(
    "Ask a question about the documents"
)


if question:

    if "vectors" not in st.session_state:

        st.warning(
            "Please click 'Create Document Embeddings' before asking a question."
        )

    else:

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using only the context below.

            If the answer cannot be found in the context,
            say that the information is not available in the document.

            <context>
            {context}
            </context>

            Question:
            {input}
            """
        )

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        retriever = st.session_state.vectors.as_retriever(
            search_kwargs={"k": 4}
        )

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        with st.spinner("Searching the document..."):

            response = retrieval_chain.invoke(
                {"input": question}
            )

        st.subheader("Answer")
        st.write(response["answer"])


        with st.expander("View Retrieved Sources"):

            for i, document in enumerate(
                response.get("context", []),
                start=1
            ):

                source = document.metadata.get(
                    "source",
                    "Unknown source"
                )

                page = document.metadata.get(
                    "page",
                    "Unknown page"
                )

                st.markdown(
                    f"**Source {i}: {source} — Page {page}**"
                )

                st.write(
                    document.page_content[:1000]
                )

                st.divider()
