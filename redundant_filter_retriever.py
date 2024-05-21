from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma
    
    def get_relevant_documents(self,query):
       emb = self.embeddings.embed_query(query) #this is the embeddings vector
        # take the embeddings and feed them into the chroma method 
       return self.chroma.max_marginal_relevance_search_by_vector(
            embedding = emb,
            lambda_mult = 0.8 #tolerance for ver y very similar topics
        )
    
    async def aget_relevant_documents(self):
        return[]