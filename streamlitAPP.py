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
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .info-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .source-card {
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">📄 Document Intelligence Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload PDF documents and ask grounded questions using NVIDIA NIM,
    LangChain, FAISS, and Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True
)


api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error(
        "NVIDIA_API_KEY is missing. "
        "Add it to your environment before running the application."
    )
    st.stop()


llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key
)


with st.sidebar:
    st.header("About")

    st.write(
        "This application performs semantic retrieval over uploaded PDFs "
        "and generates answers grounded in the retrieved document context."
    )

    st.subheader("Technology Stack")

    st.markdown(
        """
        - NVIDIA NIM
        - Nemotron LLM
        - NVIDIA Embeddings
        - LangChain
        - FAISS
        - Streamlit
        """
    )

    st.subheader("How to use")

    st.markdown(
        """
        1. Upload one or more PDFs
        2. Create document embeddings
        3. Ask questions
        4. Review retrieved sources
        """
    )

    if st.button("Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.rerun()


uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload one or multiple PDF documents."
)


def create_vector_store(uploaded_files):

    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    for old_file in temp_dir.glob("*.pdf"):
        old_file.unlink()

    for uploaded_file in uploaded_files:
        output_path = temp_dir / uploaded_file.name

        with open(output_path, "wb") as f:
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
    st.session_state.page_count = len(documents)
    st.session_state.chunk_count = len(document_chunks)


col1, col2 = st.columns([1, 2])

with col1:

    if st.button(
        "Create Document Embeddings",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Reading documents and creating vector embeddings..."
        ):
            create_vector_store(uploaded_files)

        if "vectors" in st.session_state:
            st.success("Documents processed successfully.")


with col2:

    if "vectors" in st.session_state:

        st.markdown(
            f"""
            <div class="info-card">
            <b>Ready for questions</b><br>
            Documents: {st.session_state.document_count} |
            Pages: {st.session_state.page_count} |
            Chunks: {st.session_state.chunk_count}
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()

st.subheader("Ask your documents")

question = st.text_input(
    "Question",
    placeholder="Example: What are the key risks discussed in this document?",
    label_visibility="collapsed"
)


if question:

    if "vectors" not in st.session_state:

        st.warning(
            "Upload your PDF and create document embeddings first."
        )

    else:

        prompt = ChatPromptTemplate.from_template(
            """
            You are a document question-answering assistant.

            Answer the user's question using only the provided document
            context.

            If the answer cannot be found in the context, respond:
            "The information is not available in the uploaded documents."

            Give a clear and concise answer.

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

        with st.spinner("Searching your documents..."):

            response = retrieval_chain.invoke(
                {"input": question}
            )

        st.subheader("Answer")

        with st.chat_message("assistant"):
            st.write(response["answer"])

        context_docs = response.get("context", [])

        if context_docs:

            st.subheader("Retrieved Sources")

            for i, document in enumerate(
                context_docs,
                start=1
            ):

                source = Path(
                    document.metadata.get(
                        "source",
                        "Unknown document"
                    )
                ).name

                page = document.metadata.get(
                    "page",
                    "Unknown"
                )

                if isinstance(page, int):
                    page = page + 1

                with st.expander(
                    f"Source {i} — {source} — Page {page}"
                ):

                    st.write(
                        document.page_content[:1200]
                    )