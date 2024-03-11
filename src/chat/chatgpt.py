import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['AZURE_AI_SEARCH_KEY'] = os.getenv('AZURE_AI_SEARCH_KEY')
os.environ['AZURE_AI_SEARCH_ENDPOINT'] = os.getenv('AZURE_AI_SEARCH_ENDPOINT')
os.environ['AZURE_AI_SEARCH_INDEX_NAME'] = os.getenv('AZURE_AI_SEARCH_INDEX_NAME')

PROMPT_TEMPLATE = PromptTemplate("""Given the following question, answer it based only on the information of your knowledge base, don't search
                                 anaything on the Internet.
                                 
                                 Question:
                                 {question}
                                 """)

ConversationalRetrievalChain(
    llm=llm,
    retriever=,
    condense_question_prompt=PROMPT_TEMPLATE,
    return_source_documents=True,
    verbose=True
)

query = "Como posso operar o broker?"
result = 