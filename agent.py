import os
from datetime import datetime
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphInterrupt
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd

# Load environment variables
load_dotenv()

# --- Configuration & Clients ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.get_database("gstasks")
tasks_col = db.get_collection("tasks")
tags_col = db.get_collection("tags")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_retries=2,
)

# --- Helper Functions ---


def get_tag_map():
    """Maps UUID -> Name and Name -> UUID from the tags collection."""
    tags = list(tags_col.find({}, {"uuid": 1, "name": 1}))
    uuid_to_name = {t["uuid"]: t["name"] for t in tags}
    name_to_uuid = {t["name"]: t["uuid"] for t in tags}
    return uuid_to_name, name_to_uuid


def normalize_date(date_val):
    """Handles both BSON $date objects and ISO strings for readability."""
    if isinstance(date_val, dict) and "$date" in date_val:
        return date_val["$date"]
    return str(date_val)


# --- Tool Definitions ---


@tool
def query_tasks(query_filter: dict, tag_name: Optional[str] = None, limit: int = 50):
    """
    Queries the MongoDB task database.
    - query_filter: standard MongoDB find dict.
    - tag_name: if provided (e.g., 'candychore'), automatically finds the UUID.
    - limit: defaults to 50.
    """
    try:
        u_to_n, n_to_u = get_tag_map()

        # --- AUTO-FIX: Date String to $date Object ---
        # Detect if the LLM sent a string for scheduled_date and wrap it
        if "scheduled_date" in query_filter:
            dt_val = query_filter["scheduled_date"]
            if isinstance(dt_val, str):
                # Formats "YYYY-MM-DD" into the DB's expected ISO format
                query_filter["scheduled_date"] = {"$date": pd.to_datetime(dt_val)}
            elif isinstance(dt_val, dict) and "$date" not in dt_val:
                # Handles operators like {"$gte": "2026-03-23"}
                for op, val in dt_val.items():
                    if isinstance(val, str):
                        query_filter["scheduled_date"][op] = pd.to_datetime(val)

        # Resolve tag_name to UUID if provided
        if tag_name and tag_name in n_to_u:
            query_filter["tags"] = n_to_u[tag_name]

        results = list(tasks_col.find(query_filter).limit(limit))

        readable_results = []
        for r in results:
            # Resolve tag UUIDs to human names for the LLM context
            resolved_tags = [
                u_to_n.get(tag_uuid, tag_uuid) for tag_uuid in r.get("tags", [])
            ]

            readable_results.append(
                {
                    "_id": str(r["_id"]),
                    "name": r.get("name"),
                    "status": r.get("status"),
                    "scheduled": normalize_date(r.get("scheduled_date")),
                    "tags": resolved_tags,
                    "comment": r.get("comment"),
                    "uuid": r.get("uuid"),  # keeping original uuid for reference
                }
            )

        return readable_results if readable_results else "No tasks found."
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def modify_task(task_id: str, updates: dict):
    """
    Updates an existing task in MongoDB by its _id (string).
    Handles date formatting for 'scheduled_date' automatically.
    """
    try:
        # Normalize date if the LLM sends a simple string
        if "scheduled_date" in updates and isinstance(updates["scheduled_date"], str):
            # Maintain consistency with your EJSON format
            updates["scheduled_date"] = {
                "$date": f"{updates['scheduled_date']}T00:00:00.000Z"
            }

        result = tasks_col.update_one({"_id": ObjectId(task_id)}, {"$set": updates})
        return (
            f"Update successful for {task_id}."
            if result.modified_count > 0
            else "No changes made (task not found or data identical)."
        )
    except Exception as e:
        return f"Error: {str(e)}"


tools = [query_tasks, modify_task]
tool_node = ToolNode(tools)

# --- Graph Nodes ---


def call_model(state: MessagesState):
    """Standard node to get model response."""
    current_time = datetime.now().strftime("%A, %B %d, %Y")
    system_prompt = (
        f"You are Alex's task assistant. Current date: {current_time}. "
        "When Alex asks for tasks by tag (e.g., 'chores', 'niw', 'candychore'), "
        "pass that tag string into the 'tag_name' parameter of query_tasks. "
        "Always refer to tasks by their _id when modifying them."
    )

    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    return {"messages": [response]}


def human_approval_node(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "modify_task":
                raise GraphInterrupt(
                    f"CRITICAL: Approve modification? {tool_call['args']}"
                )
    return state


# --- Routing Logic ---


def should_continue(state: MessagesState) -> Literal["action", "approval", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return END

    if any(tc["name"] == "modify_task" for tc in last_message.tool_calls):
        return "approval"

    return "action"


# --- Graph Construction ---

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("action", tool_node)
builder.add_node("approval", human_approval_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"action": "action", "approval": "approval", "__end__": END},
)
builder.add_edge("approval", "action")
builder.add_edge("action", "agent")

compile_kwargs = {}

# --- Persistence ---
if not os.getenv("IS_LANGGRAPH_DEV", "1") == "1":
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    compile_kwargs["checkpointer"] = memory

graph = builder.compile(**compile_kwargs)
