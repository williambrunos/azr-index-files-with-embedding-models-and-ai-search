from typing import (
    Callable,
    Union
)
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.azuresearch import AzureSearch

class AzureSearchClient:
    
    def __init__(
        self, 
        azure_search_endpoint: str,
        azure_search_key: str,
        index_name: str,
        embedding_function: Union[Callable, Embeddings]
        ):
        self.embedding_function = embedding_function
        self.azure_search_endpoint = azure_search_endpoint
        self.azure_search_key = azure_search_key
        self.index_name = index_name
        
            
    def _get_client(self):
        print(f'index name: {self.index_name}')
        return AzureSearch(
                azure_search_endpoint=self.azure_search_endpoint,
                azure_search_key=self.azure_search_endpoint,
                index_name=self.index_name,
                embedding_function=self.embedding_function
            )