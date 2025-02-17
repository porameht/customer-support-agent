from mongodb import client, DB_NAME
from langchain.memory import ConversationBufferMemory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import os

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        os.getenv("MONGODB_ATLAS_URI"), session_id, database_name=DB_NAME, collection_name="history"
    )


memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=get_session_history("latest_agent_session")
)
     