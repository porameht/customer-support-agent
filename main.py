from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
# from IPython.display import display, Image
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agent_memory import memory as conversation_memory
from retriever_tool import retriever_tool
from langgraph.checkpoint.memory import MemorySaver
import os

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# llm = ChatAnthropic(
#     temperature=0,
#     anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
#     model=os.getenv("MODEL"),
# )

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-2.0-flash-001",
)

def categorize(state: State) -> State:
    """Categorize the query."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize this query into 'Technical', 'Billing', 'General', or 'Package'. "
        "Respond ONLY with the category name in uppercase. "
        "Query: {query}"
    )
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content.strip().capitalize()
    valid_categories = {"Technical", "Billing", "General", "Package"}
    return {"category": category if category in valid_categories else "General"}

def analyze_sentiment(state: State) -> State:
    """Analyze sentiment of the query."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query "
        "Response with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | llm
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """Handle technical queries."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    """Handle billing queries."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    """Handle general queries."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_package(state: State) -> State:
    """Handle package queries."""
    
    # Get relevant documents using the retriever and process them
    relevant_docs = retriever_tool(state["query"])
    context = "\n".join(doc.page_content for doc in relevant_docs)
    
    chat_history = conversation_memory.load_memory_variables({})["chat_history"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are a customer service agent. Provide a helpful response about our available packages in Thai language.\n"
        "Customer query: {query}\n\n"
        "Previous conversation history: {chat_history}\n\n"
        "Additional context: {context}\n\n"
        "Provide a detailed response about our packages and help the customer choose "
        "the most suitable option based on their query. Focus on the number of Facebook Pages "
        "they can connect and highlight the 24/7 admin support available in all packages."
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "chat_history": chat_history,
        "context": context
    }).content
    
    # Save to conversation memory
    conversation_memory.save_context({"input": state["query"]}, {"output": response})
    
    return {"response": response}

def process_retriever_results(state: State) -> State:
    """Process retriever results and add them to state."""
    docs = retriever_tool(state["query"])
    context = "\n".join(doc.page_content for doc in docs)
    return {"query": state["query"], "category": state["category"], "context": context}

def escalate(state: State) -> State:
    """Escalate negative sentiment queries."""
    return {"response": "ขออภัยค่ะ คุณสามารถติดต่อเราได้ที่ 02-123-4567"}

def route_query(state: State) -> str:
    """Route query based on category and sentiment."""
    print(f"DEBUG - Category: {state['category']}, Sentiment: {state['sentiment']}")
    
    if state["sentiment"] == "Negative":
        return "escalate"
    # Use direct string comparison with normalized category
    category = state["category"].strip().capitalize()
    if category == "Package":
        return "handle_package"
    elif category == "Technical":
        return "handle_technical"
    elif category == "Billing":
        return "handle_billing"
    else:  # Fallback to general for any unexpected categories
        return "handle_general"

workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("handle_package", handle_package)
workflow.add_node("escalate", escalate)
workflow.add_node("process_retriever", process_retriever_results)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query, {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "handle_package": "handle_package",
        "escalate": "escalate"
    }
)

workflow.add_edge("process_retriever", "handle_package")

workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("handle_package", END)
workflow.add_edge("escalate", END)

workflow.set_entry_point("categorize")

checkpoint_memory = MemorySaver()

graph = workflow.compile(
    checkpointer=checkpoint_memory
)