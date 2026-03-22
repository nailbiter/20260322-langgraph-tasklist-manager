import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# --- Configuration & Clients ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.get_database("gstasks")
tasks_col = db.get_collection("tasks")

llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-flash",
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_retries=2,
)

# --- Tool Definitions ---


@tool
def query_tasks(query_filter: dict):
    """
    Queries the MongoDB task database.
    Use this for read-only operations like listing tasks.
    """
    try:
        # FIXME: why limit 20 here?
        results = list(tasks_col.find(query_filter).limit(20))
        for r in results:
            r["_id"] = str(r["_id"])
        return results if results else "No tasks found."
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def modify_task(task_id: str, updates: dict):
    """
    Updates an existing task in MongoDB.
    Requires human approval before execution.
    """
    try:
        from bson import ObjectId

        result = tasks_col.update_one({"_id": ObjectId(task_id)}, {"$set": updates})
        return (
            f"Update successful for {task_id}."
            if result.modified_count > 0
            else "No changes made."
        )
    except Exception as e:
        return f"Error: {str(e)}"


tools = [query_tasks, modify_task]
tool_node = ToolNode(tools)

# --- Graph Nodes ---


def call_model(state: MessagesState):
    """Standard node to get model response."""
    # Inject current time so the agent understands 'today' or 'next week'
    current_time = datetime.now().strftime("%A, %B %d, %Y")
    system_prompt = f"You are a task assistant. Current date: {current_time}."

    model_with_tools = llm.bind_tools(tools)
    # Prefixing messages with system context
    response = model_with_tools.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    return {"messages": [response]}


def human_approval_node(state: MessagesState):
    """
    This node only triggers for 'modify_task'.
    In LangGraph Studio, it creates a 'Breakpoint'.
    """
    last_message = state["messages"][-1]

    # We double-check here just to be safe, though the edge handles routing
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "modify_task":
            # The IDE will catch this and wait for user 'Resume'
            raise GraphInterrupt(f"CRITICAL: Approve modification? {tool_call['args']}")

    return state


# --- Routing Logic ---


def should_continue(state: MessagesState) -> Literal["action", "approval", "__end__"]:
    """
    Determines the next step based on the tool being called.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return END

    # Logic: If ANY call is a modification, we must go to approval
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "modify_task":
            return "approval"

    # If it's just 'query_tasks' (or any other read-only tool), go straight to action
    return "action"


# --- Graph Construction ---

builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("action", tool_node)
builder.add_node("approval", human_approval_node)

builder.add_edge(START, "agent")

# Route based on tool type
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"action": "action", "approval": "approval", "__end__": END},
)

# After approval, proceed to action
builder.add_edge("approval", "action")

# After action, return to agent to explain the result
builder.add_edge("action", "agent")

compile_kwargs = {}

# --- Persistence ---
if not os.getenv("IS_LANGGRAPH_DEV", "1") == "1":
    from langgraph.checkpoint.memory import MemorySaver

    # MemorySaver is the simplest in-memory checkpointer
    # for local development and IDE testing.
    memory = MemorySaver()
    compile_kwargs["checkpointer"] = memory

# Compile the graph
# The 'checkpointer' enables persistent memory across turns
graph = builder.compile(**compile_kwargs)
