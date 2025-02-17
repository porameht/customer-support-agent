from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

__all__ = ["embedding_model"]

if __name__ == "__main__":
    print(embedding_model.embed_documents(["Hello, world!"]))