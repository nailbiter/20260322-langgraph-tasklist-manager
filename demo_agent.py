import os
import uuid
from datetime import datetime
from typing import Annotated, TypedDict, Literal, List, Optional, Any, Union
from string import Template
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from common.db_interaction import (
    fetch_mongo_tasks,
    insert_task,
    update_task_by_uuid,
    get_tag_map,
    parse_date_flexible,
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
    scheduled_date: Optional[Union[dict, str]]
    URL: Optional[str]
    tags: List[str]
    due: Optional[str]
    comment: Optional[Any]


def merge_tasks(existing: List[Task], updates: List[Task]) -> List[Task]:
    """
    Pure reducer for 'tasks'. Ensures no mutation and stable sorting.
    """
    # Create a fresh map from existing tasks
    task_map = {t["uuid"]: dict(t) for t in (existing or [])}

    for ut in updates:
        if "uuid" in ut:
            tid = ut["uuid"]
            if tid in task_map:
                task_map[tid].update(ut)
            else:
                task_map[tid] = dict(ut)

    # Sort deterministically by date so the LLM prompt is stable
    def sort_key(x):
        d = parse_date_flexible(x.get("scheduled_date"))
        return d if d is not None else datetime(9999, 12, 31)

    return sorted(task_map.values(), key=sort_key)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tasks: Annotated[List[Task], merge_tasks]


# --- 2. Tool Definitions ---


@tool
def add_task(
    name: str,
    scheduled_date: Optional[str] = None,
    url: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """
    Creates a new task in MongoDB.
    scheduled_date should be 'YYYY-MM-DD'.
    tags should be a list of human-readable tag names.
    """
    new_task: Task = {
        "uuid": str(uuid.uuid4()),
        "name": name,
        "URL": url,
        "tags": tags or [],
        "scheduled_date": {"$date": f"{scheduled_date}T00:00:00.000Z"}
        if scheduled_date
        else None,
        "due": None,
        "status": "OPEN",
    }
    insert_task(new_task.copy())
    return new_task


@tool
def update_task(task_uuid: str, updates: dict):
    """
    Updates a task in MongoDB by UUID.
    Can update 'name', 'status' (DONE, CANCELLED, etc.), 'scheduled_date' (YYYY-MM-DD), 'tags', or 'comment'.
    """
    # if "scheduled_date" in updates and isinstance(updates["scheduled_date"], str):
    #     updates["scheduled_date"] = parse_date_flexible(updates["scheduled_date"])

    update_task_by_uuid(task_uuid, updates)
    updates["uuid"] = task_uuid
    return updates


@tool
def sync_from_db():
    """
    Fetches the absolute latest tasks from the database.
    Use this if the user mentions they've updated tasks elsewhere
    or if you need to ensure the most current data before a complex operation.
    """
    latest_tasks = fetch_mongo_tasks(limit=100)
    return latest_tasks


tools = [add_task, update_task, sync_from_db]

# --- 3. Nodes ---


def agent_node(state: AgentState):
    tasks = state.get("tasks", [])
    if not tasks:
        tasks = fetch_mongo_tasks()

    u_to_n, _ = get_tag_map()
    available_tags = list(u_to_n.values())

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    model_with_tools = llm.bind_tools(tools)

    current_time = datetime.now().strftime("%A, %B %d, %Y")
    system_prompt_content = PROMPT_TEMPLATE.substitute(
        current_time=current_time, tasks=tasks, available_tags=available_tags
    )

    messages = [SystemMessage(content=system_prompt_content)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "tasks": tasks}


def action_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    task_updates = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]

        res = None
        if tool_name == "add_task":
            res = add_task.invoke(args)
            task_updates.append(res)
        elif tool_name == "update_task":
            res = update_task.invoke(args)
            task_updates.append(res)
        elif tool_name == "sync_from_db":
            res = sync_from_db.invoke({})
            task_updates.extend(res)

        tool_messages.append(
            ToolMessage(
                content=f"Confirmed updated state: {res}", tool_call_id=tool_call["id"]
            )
        )

    return {"messages": tool_messages, "tasks": task_updates}


# --- 4. Logic & Graph Construction ---


def should_continue(state: AgentState) -> Literal["action", "read_action", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END

    if any(tc["name"] in ["add_task", "update_task"] for tc in last_message.tool_calls):
        return "action"

    return "read_action"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("action", action_node)
builder.add_node("read_action", action_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"action": "action", "read_action": "read_action", "__end__": END},
)
builder.add_edge("action", "agent")
builder.add_edge("read_action", "agent")

compile_kwargs = {"interrupt_before": ["action"]}

if not os.getenv("IS_LANGGRAPH_DEV", "1") == "1":
    checkpointer = MemorySaver()
    compile_kwargs["checkpointer"] = checkpointer

graph = builder.compile(**compile_kwargs)
