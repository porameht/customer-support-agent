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
        "ตรวจสอบว่าข้อความต่อไปนี้มีคำหยาบคายหรือไม่ "
        "ตอบเพียง 'Negative' ถ้ามีคำหยาบ หรือ 'Neutral' ถ้าไม่มีคำหยาบ "
        "ข้อความ: {query}"
    )
    chain = prompt | llm
    sentiment = chain.invoke({"query": state["query"]}).content.strip()
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """Handle technical queries."""
    prompt = ChatPromptTemplate.from_template(
        "คุณเป็นพนักงานฝ่ายสนับสนุนด้านเทคนิค ของ MyOrder กรุณาตอบคำถามต่อไปนี้ด้วยภาษาไทย: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    """Handle billing queries."""
    prompt = ChatPromptTemplate.from_template(
        "คุณเป็นพนักงานฝ่ายการเงิน ของ MyOrder กรุณาตอบคำถามเกี่ยวกับการชำระเงินต่อไปนี้ด้วยภาษาไทย: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    """Handle general queries."""
    prompt = ChatPromptTemplate.from_template(
        "คุณเป็นพนักงานฝ่ายบริการลูกค้า ของ MyOrder กรุณาตอบคำถามทั่วไปต่อไปนี้ด้วยภาษาไทย: {query}"
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
        "คุณเป็นพนักงานฝ่ายบริการลูกค้า ของ MyOrder ให้ข้อมูลเกี่ยวกับแพ็คเกจที่มีให้บริการ\n"
        "คำถามจากลูกค้า: {query}\n\n"
        "ประวัติการสนทนา: {chat_history}\n\n"
        "ข้อมูลเพิ่มเติม: {context}\n\n"
        "กรุณาให้รายละเอียดเกี่ยวกับแพ็คเกจของเราและช่วยลูกค้าเลือกแพ็คเกจที่เหมาะสมที่สุด "
        "โดยเน้นจำนวน Facebook Pages ที่สามารถเชื่อมต่อได้ และการมีทีมแอดมินดูแล 24 ชั่วโมงในทุกแพ็คเกจ"
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
    
    sentiment = state["sentiment"].strip().title()
    if sentiment == "Negative":
        return "escalate"

    category = state["category"].strip().capitalize()
    if category == "Package":
        return "handle_package"
    elif category == "Technical":
        return "handle_technical"
    elif category == "Billing":
        return "handle_billing"
    else:
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

workflow.set_entry_point("categorize")

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("process_retriever", "handle_package")
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("handle_package", END)
workflow.add_edge("escalate", END)

checkpoint_memory = MemorySaver()

graph = workflow.compile(
    checkpointer=checkpoint_memory
)