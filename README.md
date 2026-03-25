Document Question & Answering using NVIDIA NIM and Large Language Models (LLMs)

This repository hosts the implementation of a document-based question and answering system utilizing NVIDIA Inference Microservices (NIM) and large language models (LLMs). The system is designed to parse documents and provide answers to queries based on the content, leveraging the power of NVIDIA NIM for optimized inference and LLMs for understanding and generating human-like responses.

#### Prerequisites
To run this project, you will need:
- Python 3.7 or later
- pip (Python package installer)

#### Installation
To install the required Python packages, run the following command in your terminal:
```
pip install -r requirements.txt
```

#### Configuration
- Rename the `.env.example` file to `.env` and update it with the necessary configuration values such as API keys and database URIs if applicable.

#### Structure
- `nvidia/`: This directory contains modules specific to NVIDIA NIM integration.
- `streamlitAPP.py`: A Streamlit application that provides a user interface for interacting with the Q&A system.
- `requirements.txt`: Contains all the necessary Python packages needed to run the application.

#### Running the Application
To launch the Streamlit application, execute the following command:
```
streamlit run streamlitAPP.py
```
This command will start the Streamlit server and the application can be accessed through a web browser at the address shown in the terminal, typically `http://localhost:8501`.

#### Usage
Once the application is running, users can upload documents and enter questions. The system will process the documents using the configured LLMs and NVIDIA NIM to provide relevant answers.
