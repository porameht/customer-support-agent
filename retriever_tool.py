from mongodb import client, DB_NAME, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME
from langchain_mongodb import MongoDBAtlasVectorSearch
from model import embedding_model
import os
from langchain.tools import tool

@tool
def retriever_tool():
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=os.getenv("MONGODB_ATLAS_URI"),
        namespace=f"{DB_NAME}.{COLLECTION_NAME}",
        embedding=embedding_model,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        text_key="embedding_text",
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return retriever

if __name__ == "__main__":
    retriever = retriever_tool()
    print(retriever.get_relevant_documents("มี package อะไรบ้าง"))