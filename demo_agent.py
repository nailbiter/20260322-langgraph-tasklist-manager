import os
import uuid
from datetime import datetime
from typing import Annotated, TypedDict, Literal, List, Optional, Any, Union
from string import Template
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from common.db_interaction import (
    fetch_mongo_tasks, 
    insert_task, 
    update_task_by_uuid
)

load_dotenv()

# --- 0. Load Prompt Template ---
with open("system_prompt.template.md", "r") as f:
    PROMPT_TEMPLATE = Template(f.read())

# --- 1. State Schema (Aligned with Mongo Export) ---

class Task(TypedDict, total=False):
    uuid: str
    name: str
    status: Optional[str]
    scheduled_date: Optional[dict]
    URL: Optional[str]
    tags: List[str]
    due: Optional[str]
    comment: Optional[Any]

def merge_tasks(existing: List[Task], updates: List[Task]) -> List[Task]:
    if not existing:
        existing = []
    task_map = {t["uuid"]: t for t in existing}
    for ut in updates:
        if ut["uuid"] in task_map:
            task_map[ut["uuid"]].update(ut)
        else:
            task_map[ut["uuid"]] = ut
    return list(task_map.values())

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tasks: Annotated[List[Task], merge_tasks]

# --- 2. Tool Definitions ---

@tool
def add_task(name: str, scheduled_date: Optional[str] = None, url: Optional[str] = None):
    """
    Creates a new task in MongoDB. 
    scheduled_date should be 'YYYY-MM-DD'.
    """
    new_task: Task = {
        "uuid": str(uuid.uuid4()),
        "name": name,
        "URL": url,
        "tags": [],
        "scheduled_date": {"$date": f"{scheduled_date}T00:00:00.000Z"} if scheduled_date else None,
        "due": None,
        "status": "OPEN"
    }
    # Persist to MongoDB using common helper
    insert_task(new_task.copy())
    return new_task

@tool
def update_task(task_uuid: str, updates: dict):
    """
    Updates a task in MongoDB by UUID. 
    Can update 'name', 'status' (DONE, CANCELLED, etc.), 'scheduled_date' (YYYY-MM-DD), or 'comment'.
    """
    if "scheduled_date" in updates and isinstance(updates["scheduled_date"], str):
        updates["scheduled_date"] = {"$date": f"{updates['scheduled_date']}T00:00:00.000Z"}
    
    # Persist to MongoDB using common helper
    update_task_by_uuid(task_uuid, updates)
    updates["uuid"] = task_uuid
    return updates

tools = [add_task, update_task]

# --- 3. Nodes ---

def agent_node(state: AgentState):
    tasks = state.get("tasks", [])
    if not tasks:
        tasks = fetch_mongo_tasks()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    model_with_tools = llm.bind_tools(tools)
    
    current_time = datetime.now().strftime("%A, %B %d, %Y")
    system_prompt = PROMPT_TEMPLATE.substitute(
        current_time=current_time,
        tasks=tasks
    )
    
    messages = [HumanMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "tasks": tasks}

def action_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    task_updates = []
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "add_task":
            res = add_task.invoke(tool_call["args"])
            task_updates.append(res)
        elif tool_call["name"] == "update_task":
            res = update_task.invoke(tool_call["args"])
            task_updates.append(res)
        
        tool_messages.append(ToolMessage(
            content=f"DB Operation Success: {tool_call['args']}",
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": tool_messages, "tasks": task_updates}

# --- 4. Logic & Graph Construction ---

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("action", action_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", 
    lambda s: "action" if s["messages"][-1].tool_calls else END
)
builder.add_edge("action", "agent")

compile_kwargs = {
    "interrupt_before": ["action"]
}

# Only add a custom checkpointer if NOT running in LangGraph Studio/CLI environment
if not os.getenv("IS_LANGGRAPH_DEV", "1") == "1":
    checkpointer = MemorySaver()
    compile_kwargs["checkpointer"] = checkpointer

graph = builder.compile(**compile_kwargs)
