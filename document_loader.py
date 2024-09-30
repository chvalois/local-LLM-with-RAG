import pandas as pd
import logging

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from process_documents import extract_elements_from_pdf
import os

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)


def load_documents_into_database(model_name, documents_path, document_type) -> Chroma:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """

    #persist_directory = f"./chroma_db/chroma_db_{documents_path.replace("\\", "_")}"
    persist_directory = f"./chroma_db/chroma_db_{documents_path}"

    # Vérifier si une base de données persistante existe déjà
    if os.path.exists(persist_directory):
        print("Chargement de la base de données Chroma existante")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=OllamaEmbeddings(model=model_name)
        )
    else:
        print(f"Création d'une nouvelle base de données Chroma dans {persist_directory}")

        # List all PDF files
        if document_type == 'Autres documents':
            
            print("Chargement classique de docuemnts")
            raw_documents = load_documents(documents_path)
            documents = TEXT_SPLITTER.split_documents(raw_documents)

        else:
            print("Extraction de PDFs")            
            documents = extract_elements_from_pdf(documents_path)

        print("Chargement des documents dans la base ChromaDB")

        try:
            db = Chroma.from_documents(
            documents,
            OllamaEmbeddings(model=model_name),
            persist_directory=persist_directory
            )
            
            logging.info("Documents loaded successfully")
            print("Documents loaded successfully")

        except Exception as e:
            logging.error(f"Error loading documents: {e}")
            print(f"Error loading documents: {e}")

        db = Chroma.from_documents(
        documents,
        OllamaEmbeddings(model=model_name),
        persist_directory=persist_directory
        )

        db.persist()
    

    return db


def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified directory path.

    This function supports loading of PDF, Markdown, and HTML documents by utilizing
    different loaders for each file type. It checks if the provided path exists and
    raises a FileNotFoundError if it does not. It then iterates over the supported
    file types and uses the corresponding loader to load the documents into a list.

    Args:
        path (str): The path to the directory containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    path = os.path.join("sources", path)
    
    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
#            loader_cls=loader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs
