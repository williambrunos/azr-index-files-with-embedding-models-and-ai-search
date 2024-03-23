import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.vectorstores.azuresearch import AzureSearch

# from src.utils.azuresearchclient import AzureSearchClient

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['AZURE_AI_SEARCH_KEY'] = os.getenv('AZURE_AI_SEARCH_KEY')
os.environ['AZURE_AI_SEARCH_ENDPOINT'] = os.getenv('AZURE_AI_SEARCH_ENDPOINT')
os.environ['AZURE_AI_SEARCH_INDEX_NAME'] = os.getenv('AZURE_AI_SEARCH_INDEX_NAME')

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0, 
    openai_api_key=os.environ['OPENAI_API_KEY']
    )

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    chunk_size=1,
    openai_api_key=os.environ['OPENAI_API_KEY'],
    show_progress_bar=True
)

print(os.environ['AZURE_AI_SEARCH_INDEX_NAME'])
print(os.environ['AZURE_AI_SEARCH_ENDPOINT'])

azure_ai_search = AzureSearch(
    azure_search_endpoint=os.environ['AZURE_AI_SEARCH_ENDPOINT'],
    azure_search_key=os.environ['AZURE_AI_SEARCH_KEY'],
    index_name=os.environ['AZURE_AI_SEARCH_INDEX_NAME'],
    embedding_function=embeddings.embed_query
)

loader_kwargs = {'autodetect_encoding': True}
loader = DirectoryLoader('../data/', glob='*.txt', loader_cls=TextLoader, loader_kwargs=loader_kwargs)

documents = loader.load()

azure_ai_search.add_documents(documents=documents)