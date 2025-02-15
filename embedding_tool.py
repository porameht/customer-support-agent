from mongodb import client, collection
import asyncio
from model import embedding_model
from langchain.tools import tool

@tool
async def embedding_tool(text: str, metadata: dict = None):
    try:
        client.admin.command('ping')
        print("Connected to MongoDB successfully")
        
        document = {
            "embedding_text": text,
            "metadata": metadata or {},
            "embedding": embedding_model.embed_documents([text])[0]
        }

        collection.insert_one(document)
        print("Embedding saved successfully")
        return True

    except Exception as error:
        print("Error saving embedding:", error)
        return False
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(embedding_tool("Hello, world!"))