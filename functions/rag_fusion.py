
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.load import dumps, loads

from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Multi-Query Generation & Re-ranking
template_EN = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

template_FR = """Tu es un assistant expert qui doit générer 3 réponses différentes à la question de l'utilisateur 
en t'appuyant sur les documents les plus pertinents de la base de données de vecteurs : {context}

Chaque réponse doit être séparée par un retour à la ligne \n
Voici la question de l'utilisateur à laquelle tu dois répondre : {question} \n
Résultat (3 réponses):
"""


# GENERATION
prompt_template_EN = """Answer the following question based on this context:
{context}
Question: {question}
"""

prompt_template_FR = """Réponds à la question en te basant sur ce contexte:
{context}
Question: {question}
"""

# https://gist.github.com/srcecde/eec6c5dda268f9a58473e1c14735c7bb
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse = True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


# Query
def query(query, memory, llm, db, language):

    if language == 'EN':
        prompt_rag_fusion = ChatPromptTemplate.from_template(template_EN)
        prompt = ChatPromptTemplate.from_template(prompt_template_EN)

    elif language == 'FR':
        prompt_rag_fusion = ChatPromptTemplate.from_template(template_FR)
        prompt = ChatPromptTemplate.from_template(prompt_template_FR)

    db_retriever = db.as_retriever(search_kwargs={"k": 10})
    retrieved_docs = db_retriever.get_relevant_documents(query)
    
    # Modification de la témperature du modèle afin de générer des réponses différentes
    original_temperature = llm.temperature
    llm.temperature = 0

    retrieval_chain_rag_fusion = (
        prompt_rag_fusion
        | llm
        | StrOutputParser() 
    ).with_config({"run_name": "Multi-réponse"})

    llm.temperature = original_temperature

    docs = retrieval_chain_rag_fusion.invoke({"question": query, "context": retrieved_docs})

    print(f"{Fore.CYAN}{len(docs)} documents retrieved {Fore.RESET}")
    
    # GENERATION Chain
    final_rag_chain = (
        {"context": lambda x: docs, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    ).with_config({"run_name": "Final Answer"})

    return final_rag_chain.stream({"question": query, "context": final_rag_chain})


    