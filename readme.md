# Local LLM with RAG

This project is an experimental sandbox for testing out ideas related to running local Large Language Models (LLMs) with [Ollama](https://ollama.ai/) to perform Retrieval-Augmented Generation (RAG) for answering questions based on sample PDFs. 

In this project, we are also using Ollama to create embeddings with the [nomic-embed-text](https://ollama.com/library/nomic-embed-text) to use with [Chroma](https://docs.trychroma.com/). 

There is also a web UI created using [Streamlit](https://streamlit.io/) to provide a different way to interact with Ollama.

<p align="center">
    <img src="images/streamlit_ui.png" alt="Screenshot of Streamlit web UI" width="600">
</p>

## Requirements

- [Ollama](https://ollama.ai/) verson 0.1.26 or higher.

## Setup

To run it locally : 
1. Clone this repository to your local machine.
2. Create a Python virtual environment by running `python3 -m venv .venv`.
3. Activate the virtual environment by running `source .venv/bin/activate` on Unix or MacOS, or `.\.venv\Scripts\activate` on Windows.
4. Install the required Python packages by running `pip install -r requirements.txt`.

To run it with Docker : 
```
docker build . -t myllm:latest
docker-compose up -d
```

## Running the Streamlit UI

1. Ensure your virtual environment is activated.
2. Navigate to the directory containing the `ui.py` script.
3. Run the Streamlit application by executing `streamlit run ui.py` in your terminal.

This will start a local web server and open a new tab in your default web browser where you can interact with the application. The Streamlit UI allows you to select models, select a folder, providing an easier and more intuitive way to interact with the RAG chatbot system compared to the command-line interface. The application will handle the loading of documents, generating embeddings, querying the collection, and displaying the results interactively.

## Technologies Used

- [Langchain](https://github.com/langchain/langchain): A Python library for working with Large Language Model
- [Ollama](https://ollama.ai/): A platform for running Large Language models locally.
- [Chroma](https://docs.trychroma.com/): A vector database for storing and retrieving embeddings.
- [PyPDF](https://pypi.org/project/PyPDF2/): A Python library for reading and manipulating PDF files.
- [Streamlit](https://streamlit.io/): A web framework for creating interactive applications for machine learning and data science projects.
