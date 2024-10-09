
from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.load import dumps, loads
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_mistralai import ChatMistralAI

import os
from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


#### Retrieval Grader : Retrieval Evaluator ####
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    def get_score(self) -> str:
        """Return the binary score as a string."""
        return self.binary_score


def get_score(self) -> str:
    """Return the binary score as a string."""
    return self.binary_score


# Prompt 
system_template = """You are an evaluator determining the relevance of a retrieved documents to a user's query.
If the document contains keyword(s) or semantic meaning related to the question, mark it as relevant.
Assign a binary score of 'yes' or 'no' to indicate the document's relevance to the question.

Here is the document: {documents}
Here is the user query : {question}"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["documents", "question"],
    template="{question}",
)
grader_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

### Question Re-writer - Knowledge Refinement ####
# Prompt 
prompt_template = """Given a user input {question}, your task is re-write or rephrase the question to optimize the query in order to improve the content generation"""

system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
human_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question"],
    template="{question}",
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt]
)

### Web Search Tool - Knowledge Searching ####
web_search_tool = TavilySearchResults(k=3) 

#### Generate Answer  ####
# Prompt
prompt = hub.pull("rlm/rag-prompt")


# Retrieve and assess
def assess_retrieved_docs(query, llm, db_retriever, documents=[]):

    llm_grader = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))
    # LLM with function call 
    structured_llm_grader = llm_grader.with_structured_output(GradeDocuments)

    """Retrieve and assess the relevance of documents to a given query."""
    retrieval_grader = grader_prompt | structured_llm_grader | get_score
    docs = db_retriever.get_relevant_documents(query) 
    doc_txt = docs[0].page_content
    binary_score = retrieval_grader.invoke({"question": query, "documents": doc_txt})
    return binary_score, docs

# Rewrite and optimize 
def rewrite_query_user(query, llm):
    """Rewrite and optimize a given user query for the model."""
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter.invoke({"question": query})

# Search the web
def search_web(query):
    """Search the web for complimentary information."""
    docs = web_search_tool.invoke({"query": query})
    print(docs)
    web_results = "\n".join([d["content"] for d in docs])
    return Document(page_content=web_results)

def generate_answer(docs, query, llm):
        
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # Run
    return rag_chain.invoke({"context": docs, "question": query})

def query(query, llm, db):
    db_retriever = db.as_retriever(search_kwargs={"k": 10})

    """Query the model with a question and assess the relevance of retrieved documents."""
    # question = "RAG"
    binary_score, docs = assess_retrieved_docs(query)
    print(f"Relevance score: {binary_score}")
    # Rewrite and optimize the query
    print(f"{Fore.YELLOW}Rewriting the query for content generation.{Fore.RESET}")
    optimized_query = rewrite_query(query)
    if binary_score == "no":
        print(f"{Fore.MAGENTA}Retrieved documents are irrelevant. Searching the web for additional information.{Fore.RESET}")
        docs = search_web(optimized_query)
    return generate_answer(docs, optimized_query, llm)