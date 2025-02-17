from mongodb import client, DB_NAME, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME
from langchain_mongodb import MongoDBAtlasVectorSearch
from embedding_model import embedding_model
import os
from langchain.agents import tool

@tool
def retriever_tool(query: str):
    """Retrieves relevant documents from MongoDB Atlas Vector Search.
    
    Args:
        query: The search query string to find relevant documents.
        
    Returns:
        A list of relevant documents found using vector similarity search.
        Returns the top 5 most relevant documents.
    """
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=os.getenv("MONGODB_ATLAS_URI"),
        namespace=f"{DB_NAME}.{COLLECTION_NAME}",
        embedding=embedding_model,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        text_key="embedding_text",
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return retriever.get_relevant_documents(query)

if __name__ == "__main__":
    results = retriever_tool("มี package อะไรบ้าง")
    print(results)