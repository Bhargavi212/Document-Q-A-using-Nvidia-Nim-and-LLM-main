# 📄 Document Intelligence Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that lets users upload PDF documents and ask grounded questions about their content.

Built with **NVIDIA NIM, NVIDIA Nemotron, LangChain, FAISS, and Streamlit**.

## 🚀 Features

- Upload one or multiple PDF documents
- Automatic PDF text extraction and chunking
- NVIDIA-powered semantic embeddings
- FAISS vector similarity search
- Natural-language document Q&A
- Grounded answers using NVIDIA Nemotron
- Retrieved sources with page numbers
- Multi-document support
- Interactive Streamlit interface

## 🧠 RAG Architecture

PDF Upload → Text Extraction → Chunking → NVIDIA Embeddings → FAISS → Semantic Retrieval → NVIDIA NIM → Grounded Answer + Sources

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM API | NVIDIA NIM |
| Generation Model | NVIDIA Nemotron |
| Embeddings | NVIDIA Nemotron Embed |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| PDF Processing | PyPDF |
| Language | Python |

## 🤖 Models

**Generation:** `nvidia/nemotron-3.5-lightning-30b-a3b`

**Embeddings:** `nvidia/nemotron-3-embed-1b`

## 🔍 RAG Configuration

- Chunk size: 700
- Chunk overlap: 100
- Retrieved chunks: Top 4
- Embedding dimension: 2048

The LLM is instructed to answer only from retrieved document context. If the answer cannot be found, the application reports that the information is unavailable in the uploaded documents.

## 💡 Example

Upload the **NIST AI Risk Management Framework (AI RMF 1.0)** and ask:

> What are the core functions of the NIST AI Risk Management Framework?

Example answer:

> The core functions are GOVERN, MAP, MEASURE, and MANAGE.

Retrieved source sections and page numbers are displayed for verification.

## 📦 Installation

Clone the repository:

`git clone https://github.com/Bhargavi212/Document-Q-A-using-Nvidia-Nim-and-LLM-main.git`

Install dependencies:

`pip install -r requirements.txt`

## 🔐 NVIDIA API Key

Set the `NVIDIA_API_KEY` environment variable or place it in a local `.env` file.

**Never commit API keys or `.env` files to GitHub.**

## ▶️ Run

`streamlit run streamlitAPP.py`

## 🎯 Use Cases

- Research paper analysis
- Financial document review
- Compliance and policy Q&A
- Technical documentation search
- Academic literature analysis
- Enterprise knowledge assistants

## 🔮 Future Improvements

- Conversational memory
- Hybrid search and reranking
- Automated RAG evaluation
- Persistent vector storage
- Docker containerization
- CI/CD
- Production cloud deployment

## 👩‍💻 Author

**Bhargavi Reddy Alumolu**

GitHub: https://github.com/Bhargavi212