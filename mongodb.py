from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "cs_agent"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))

db = client[DB_NAME]
collection = db[COLLECTION_NAME]

__all__ = ['client', 'db', 'collection', 'DB_NAME', 'COLLECTION_NAME', 'ATLAS_VECTOR_SEARCH_INDEX_NAME']
