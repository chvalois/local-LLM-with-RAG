from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import bs4

from functions.document_loader import load_documents_into_database
from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# ### Construct retriever ###
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# retriever = vectorstore.as_retriever()

vectorstore = load_documents_into_database(EMBEDDING_MODEL, "src_hiring", "Documents PDF")
retriever = vectorstore.as_retriever()

# Define the prompt template
template_hr = """
You are an expert HR assistant. Based on the interview transcript provided, give a summary of the candidate skills identified so far
and a concise recommendation to the recruiter on how to proceed with the next steps. 
The transcript reflects a real-time interview, and your response should help improve the evaluation of the candidate.

Latest Interview Transcript:
{context}

Soft Skills of the candidate (list of soft skills): 
Hard skills of the candidate (list of hard skills):
Review of the interview (note out of 10, 10 is max note)
Recommendation to the recruiter (1 sentence):
"""

# Define the prompt template
template_hr_2 = """
You are an expert HR assistant. 
Based on the interview transcript provided, give a summary of the candidate skills identified so far and a concise recommendation to the recruiter on how to proceed with the rest of the interview. 
The transcript reflects a real-time interview, and your response should help the recruiter ask the perfect question to continue evaluating the candidate.

Latest Interview Transcript:
{context}

Soft Skills of the candidate (list of soft skills): 
Hard skills of the candidate (list of hard skills):
Review of the interview (note out of 10, 10 is max note)
Recommendation to the recruiter (1 sentence):
"""

# Define the prompt template
template_strategic = """
You are an expert in Agile coaching.
Based on the interview transcript provided and the knowledge of the vector database, give a recommendation to the participants about how to keep on holding the meeting.
Talk to them directly.

Latest Meeting Transcript:
{context}

Recommendation to the participants (1 sentence):
"""

# Example usage with transcript
transcript_hr = """
Transcript: Recruitment Interview for Data Analyst Position

Date: September 9, 2024 Duration: 30 minutes Interviewer: Jane Smith, Senior Data Analyst Candidate: Alex Johnson

Jane Smith: Good morning, Alex. Thank you for joining us today. How are you?

Alex Johnson: Good morning, Jane. I'm doing well, thank you. How are you?

Jane Smith: I'm great, thanks for asking. Let's dive right in. I see from your resume that you have a strong background in data analysis and statistical methods. 
Could you start by telling me a bit about your most recent role and some of the projects you've worked on?

Alex Johnson: Sure. In my previous role at XYZ Corp, I was involved in several projects that required extensive data analysis. 
For instance, I worked on a project analyzing customer behavior to improve our marketing strategies. 
I used Python and SQL for data extraction and cleaning, and applied various statistical models to identify key trends and insights.

Jane Smith: That sounds interesting. Could you give me an example of a challenge you faced during that project and how you addressed it?

Alex Johnson: One challenge was dealing with incomplete and inconsistent data from different sources. 
To address this, I implemented a data validation process to ensure the accuracy of the data before performing any analysis. 
I also created a detailed documentation of the data cleaning procedures, which helped the team understand the adjustments made and maintain consistency.

Jane Smith: Excellent. It’s clear you have strong technical skills. 
However, we’re also looking for someone who can communicate their findings effectively to non-technical stakeholders. 
Can you provide an example of how you've done this in the past?

Alex Johnson: Yes, in my previous role, I was responsible for presenting the findings of my analysis to the marketing team. 
I created visualizations and summary reports to highlight key insights in a way that was easy for them to understand. 
I also made sure to explain the methodology in simple terms and answered any questions they had to ensure they were comfortable with the findings.

Jane Smith: Great. It's important for us to ensure that our team members can translate complex data into actionable insights. 
Lastly, before we move on, do you have any questions about the role or our company?

Alex Johnson: I do have a few questions. Could you tell me more about the type of projects the data analyst team typically works on? 
And how does the team collaborate with other departments?


Next 5 minutes of the meeting : 

Jane Smith: Thank you for being open to that idea, Alex. 
To give you a sense of what we’re looking for, I’ll present you with a brief case study scenario. 
Imagine that you’re given a dataset from a recent customer satisfaction survey. 
The dataset includes customer feedback, ratings, and demographic information. 
Your task is to identify any significant trends and provide actionable recommendations to improve customer satisfaction. How would you approach this?

Alex Johnson: Okay, I’d start by performing an initial data exploration to understand the dataset’s structure and identify any potential issues, such as missing values or outliers. 
Next, I’d clean the data to ensure its accuracy and consistency. 
Then, I’d use statistical analysis and data visualization techniques to uncover patterns and trends in the feedback and ratings. 
For instance, I might use clustering to segment customers and identify specific areas where satisfaction is low. 
Finally, I’d compile the findings into a report with clear, actionable recommendations for improving customer satisfaction.

Jane Smith: That’s a solid approach. Could you elaborate on the types of statistical analysis or visualizations you might use for this scenario?

Alex Johnson: Sure. For statistical analysis, I might use techniques like correlation analysis to explore relationships between different variables, 
or regression analysis to predict factors influencing customer satisfaction. 
For visualizations, I’d create charts and graphs such as bar charts, heatmaps, and pie charts to illustrate key trends and comparisons. 
Tools like Tableau or Matplotlib in Python could be useful for this purpose.

Jane Smith: Great, it sounds like you have a strong grasp of both the technical and analytical aspects. 
Now, let’s shift gears a bit. Given that our team often works under tight deadlines and in a fast-paced environment, 
how do you prioritize and manage your tasks to ensure timely and accurate results?

Alex Johnson: In a fast-paced environment, I prioritize tasks based on their impact and urgency. 
I typically start by breaking down larger projects into smaller, manageable tasks and set deadlines for each one. 
I also use project management tools to track progress and communicate with the team. 
Regular check-ins and updates help ensure that everyone is aligned and any potential issues are addressed promptly.

Jane Smith: Excellent approach. It’s important for us to have teams members who can handle pressure effectively. 
We’ve covered a lot today, and I appreciate your thoughtful responses. Do you have any more questions for me about the role or our team?
"""

# Example usage with transcript
transcript_strategic = """

Goal of the meeting : decide which projects must be prioritized and within how much time.

Strategic Meeting Transcript (First 5 Minutes)
Date: September 9, 2024
Duration: 30 minutes
Attendees:

CEO (Jane Parker)
Head of Product (Michael Davis)
Head of Marketing (Sara Martinez)
Head of Sales (David Nguyen)
Head of IT (Laura Bennett)

CEO (Jane Parker):
Good morning, everyone. I appreciate you all making time for this. As you know, the main objective today is to decide on the products we’ll focus on for development next quarter. We have a lot of great ideas in the pipeline, but we need to be strategic. Our resources are limited, so we can’t take on everything.

Michael, since you’re leading product, can you start by outlining what’s currently on the table?

Head of Product (Michael Davis):
Of course, Jane. We’ve narrowed it down to three core ideas based on market research and internal feedback:

SmartHub 2.0 – an upgrade to our flagship product with predictive analytics powered by machine learning.

MobileLite – a simplified version of our current platform targeting small businesses, especially those who don’t need all the features of the full SmartHub.

AI Assistant API – this would open up our AI capabilities to external developers, allowing them to integrate our assistant into their platforms.

Each of these products offers different opportunities, but also comes with its own challenges in terms of development and market positioning.

CEO (Jane Parker):
Thanks, Michael. Let’s get some input from the rest of the team. Sara, from a marketing perspective, which of these do you think we should prioritize, and why?

Head of Marketing (Sara Martinez):
Well, all three have strong potential, but if I had to pick, SmartHub 2.0 would be my top choice for immediate development. The reason is simple – it’s an upgrade to a product our current customers already know and love. We can run a strong campaign around its new features, and the transition would be smooth.

MobileLite is also interesting, especially for tapping into the small business market, but it might require more time to build awareness and adoption. It’s a new audience for us.

CEO (Jane Parker):
Good point, Sara. David, how do you see this from the sales side?
"""


# Define the LLM (this can be replaced with a local LLM)
llm = Ollama(model="llama3.1:8b")

prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", template_hr_2),
        ("system", template_strategic),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# response = rag_chain.invoke({"input": transcript_strategic})
# print(response['answer'])

