from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

embeddings = OpenAIEmbeddings()
chat = ChatOpenAI()   
db = Chroma (
    persist_directory = "emb",
embedding_function=embeddings,    
)

retriever = RedundantFilterRetriever(embeddings=embeddings,chroma = db)

chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type = "stuff"
)

result = chain.run("what is an interesting fact about the english language")

print(result)