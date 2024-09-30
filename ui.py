import streamlit as st
import os
import time
from dotenv import load_dotenv
import logging

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from document_loader import load_documents_into_database
from models import get_list_of_models

from llm import getStreamingChain, get_reco_transcript
from C_rag_fusion import query as rag_fusion
from E_graph import query as graph



load_dotenv()
EMBEDDING_MODEL = ""
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

st.title("🗄️ Votre LLM open-source en local")

#### ---- Logging ---- ####

logging.basicConfig(filename='app.log',   # Log file name
                    filemode='w',         # 'w' to overwrite, 'a' to append
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
                    level=logging.INFO)   # Set the log level


#### ---- Formulaire ---- ####

st.sidebar.subheader("Indexation de documents")

# Sélection du type de documents
document_type = st.sidebar.selectbox(
    "Sélectionner un type de documents : ",
    ['Documents PDF', 'Autres documents'])

# Sélection du répertoire
path = './sources/'
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('src_')]
folder_path = st.sidebar.selectbox("Sélectionner un répertoire de documents :", directories)
st.session_state["folder_path"] = ''

st.sidebar.subheader("Paramètres du Chatbot")

# Sélection du modèle
if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

selected_model = st.sidebar.selectbox(
    "Sélectionner un modèle LLM : ", 
    st.session_state["list_of_models"]
)

if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = Ollama(model=selected_model)

# Sélectionner la langue des prompts
language = st.sidebar.selectbox("Sélectionner la langue des prompts : ", ['EN', 'FR'])

# Type de réponse
answer_type = st.sidebar.selectbox("Sélectionner un type de réponse", 
                                   options = ['Basique', 'Multi', 'Open', 'Reco over transcript'], 
                                   help="Multi : la requête donne lieu à plusieurs réponses, le chatbot en fait un condensé. \
                                    Open : le chatbot va rechercher sur le web si la réponse n'est pas satisfaisante")


# Embedding des documents si la base de données de vecteurs n'existe pas
if folder_path:
    if not os.path.isdir(os.path.join("sources", folder_path)):
        st.error(
            f"Le chemin du répertoire semble incorrect. Aucun répertoire du nom de {folder_path}"
        )
    else:
        if st.sidebar.button("Indexer des documents"):
            if ("db" not in st.session_state) | (st.session_state["folder_path"] != folder_path):
                with st.spinner(
                    f"Création des embeddings et chargement des documents situés dans le répertoire '{folder_path}' dans \
                    Chroma à l'aide du modèle '{EMBEDDING_MODEL}' ..."
                ):
                    start_time = time.time()

                    st.session_state["db"] = load_documents_into_database(EMBEDDING_MODEL, folder_path, document_type)
                    st.session_state["folder_path"] = folder_path

                    end_time = time.time()
                    total_time = round(end_time - start_time, 0)

                    st.info(f"Temps nécessaire pour indexer les documents : {total_time} secondes")  
                    logging.info(f"{folder_path} | Temps nécessaire pour indexer les documents : {total_time} secondes")

            st.info("Prêt à recevoir vos questions !")

else:
    st.warning("Please enter a folder path to load documents into the database.")

#### ---- Chatbot ---- ####

# Initialiser l'historique du chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages du chat à partir de l'historique lors que l'appli est relancée
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Question"):
    
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):

        response = ""

        if answer_type == "Basique":

            # stream = getStreamingChain(prompt, st.session_state.messages, st.session_state["llm"], st.session_state["db"], language)
            # response = st.write_stream(stream)

            stream = getStreamingChain(prompt, st.session_state.messages, st.session_state["llm"], st.session_state["db"], language)
            response = st.write_stream(stream)
            

        elif answer_type == "Multi":
            stream = rag_fusion(prompt, st.session_state.messages, st.session_state["llm"], st.session_state["db"], language)
            response = st.write_stream(stream)

        elif answer_type == "Open":
            stream = graph(prompt, st.session_state["llm"], st.session_state["db"])
            response = st.write_stream(stream)

        elif answer_type == "Reco over transcript":
            stream = get_reco_transcript(prompt, st.session_state["llm"], st.session_state["db"], language)
            response = st.write_stream(stream)            

        st.session_state.messages.append({"role": "assistant", "content": response})

