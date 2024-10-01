from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

condense_question_EN = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

condense_question_FR = """
A partir de la conversation suivante, et de la question posée par l'utilisateur, reformule la question pour être une question à elle seule.

Historique de la conversation : 
{chat_history}

Question posée par l'utilisateur : {question}

Question reformulée : """


answer_EN = """
### Instruction:
You're a helpful research assistant, who answers questions based on provided research in a clear way and easy-to-understand way.
If there is no research, or the research is irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources

## Research:
{context}

## Question:
{question}
"""

answer_FR = """
### Instruction:
Tu es un assistant de recherche, qui répond aux questions basées sur les recherches fournies d'une façon claire et facile à comprendre.
S'il n'y a pas de documents fournis ou si les documents ne sont pas pertinents pour répondre à la question, réponds que tu ne sais pas répondre.
Réponds simplement avec la réponse détaillées et avec les sources. Si tu n'es pas en mesure de répondre, ne liste pas les sources.

## Recherche:
{context}

## Question:
{question}
"""

answer_FR_test = """
### Instruction:
Tu es un expert en recrutement dans l'industrie de la data, et tu as accès à la transcription d'un entretien d'embauche pour un poste
d'expert data qui correspondant au contexte. Ton rôle est de régulièrement faire un retour sur la qualité du candidat et aider le recruteur en fournissant une courte
recommandation pour rendre l'entretien encore plus efficace sur la base de la discussion et des documents à ta disposition.

Le format attendu est le suivant : 
"Soft Skills du candidat" : liste des soft skills du candidat basée sur la discussion. Juste une liste, pas de phrase.
"Hard Skills du candidat" : liste des hard skills du candidat basée sur la discussion. Juste une liste, pas de phrase.
"Qualité du candidat" : une phrase pour estimer la qualité du candidat, et une note entre 0 et 10, 10 représente une performance exceptionnelle.
"La recommandation de votre coach RH" : une courte recommandation d'une phrase correspondant à la question qu'un expert recruteur RH conseillerait au recruteur de poser, comme si la recommandation avait lieu en direct.

## Transcript de la discussion :
{context}

## Documents à disposition :
{docs}

## Recommandation RH :
"""

answer_FR_test = """
You are an expert HR assistant. Based on the interview transcript provided so far, give a concise recommendation to the recruiter on how to proceed with the next steps. 
The transcript and chat history reflect a real-time interview, and your response should help improve the evaluation of the candidate.

Chat History:
{chat_history}

Latest Interview Transcript:
{transcript}

Your Recommendation (2-3 sentences):
"""


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)


def getStreamingChain(question: str, memory, llm, db, language):

    if language == 'EN':
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_EN)
        ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_EN)

    else:
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_FR)
        ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_FR)

    retriever = db.as_retriever(search_kwargs={"k": 10})
    
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{item['role']}: {item['content']}" for item in x["memory"]]
            )
        ),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = final_inputs | ANSWER_PROMPT | llm

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain.stream({"question": question, "memory": memory})


# Prompt avec avis RH complet
template_hr_1_EN = """
You are an expert HR assistant in data industry.

What is expected from the candidates are : 
- detailed answers with solid examples
- answers using STAR methodology (Situation, Task, Action, Result)

Based on the interview transcript provided, give a summary of the candidate skills identified so far and a concise recommendation to the recruiter on how to proceed with the rest of the interview. 
The transcript reflects a real-time interview, and your response should help the recruiter ask the perfect question to continue evaluating the candidate.
Keep in mind the interview is not over yet, and your answer will help the recruiter focus for the rest of the interview, so the recommendation should be a question to be asked to the candidate.

Think step by step.

Latest Interview Transcript:
{context}

Your answer should only limit to the following list: 

- Soft Skills of the candidate (list of soft skills): 
- Hard skills of the candidate (list of hard skills):
- Review of the interview (note out of 10, 10 is max note)
- Recommendation to the recruiter (1 sentence):
"""

# Prompt avec question à poser spécialisée data
template_hr_2_EN = """
You are an expert HR assistant in data industry.
Based on the interview transcript provided, give one question the reeruiter should ask the candidate to better evaluate if the candidate is a good fit.

Latest Interview Transcript:
{context}

The question the recruiter should ask (in only one sentence):
"""

# Prompt avec question à poser sans spécialité mentionnée
template_hr_3_EN = """
You are an expert HR assistant.
Based on the interview transcript provided, give one question the reeruiter should ask the candidate to better evaluate if the candidate is a good fit.

Latest Interview Transcript:
{context}

The question the recruiter should ask (in only one sentence):
"""

def get_reco_transcript(transcript, llm, db, language):

    db_retriever = db.as_retriever(search_kwargs={"k": 10})
    template = template_hr_2_EN

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(db_retriever, question_answer_chain)

    response = rag_chain.stream({"input": transcript})
    
    for s in response:
        yield s.get("answer", "")


