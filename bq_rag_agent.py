"""
Simple RAG agent for BigQuery table discovery.

Loads table metadata from bq_metadata/*.json, indexes it with fastembed,
and lets an LLM suggest relevant tables based on a natural-language query.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------------------------------------------------------------------
# 1. Load BQ metadata and build documents (one per table)
# ---------------------------------------------------------------------------
from langchain_core.documents import Document

def _table_to_document(path: Path) -> Document:
    meta = json.loads(path.read_text())
    ref = meta["tableReference"]
    full_name = f"{ref['projectId']}.{ref['datasetId']}.{ref['tableId']}"
    description = meta.get("description", "No description available.")
    fields = meta.get("schema", {}).get("fields", [])
    columns = ", ".join(
        f"{f['name']} ({f['type']})" + (f": {f['description']}" if f.get("description") else "")
        for f in fields
    )
    text = (
        f"Table: {full_name}\n"
        f"Description: {description}\n"
        f"Columns: {columns}"
    )
    return Document(page_content=text, metadata={"table": full_name})

metadata_dir = Path("bq_metadata")
docs = [_table_to_document(p) for p in sorted(metadata_dir.glob("*.json"))
        if not p.name.startswith("thelook")]  # skip the dataset-level summary file

# ---------------------------------------------------------------------------
# 2. Index into vector store
# ---------------------------------------------------------------------------
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.tools import tool

vectorstore = InMemoryVectorStore.from_documents(
    documents=docs,
    embedding=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def search_tables(query: str) -> str:
    """Search BigQuery table metadata by natural language query.
    Returns the most relevant table names, descriptions, and schemas."""
    results = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in results)


# ---------------------------------------------------------------------------
# 3. Agent
# ---------------------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

SYSTEM_PROMPT = (
    "You are a BigQuery assistant that helps users find the right tables for their queries. "
    "Always use the search_tables tool to find relevant tables before answering. "
    "After retrieving results, suggest which tables to use and briefly explain why."
)


def agent_node(state: MessagesState):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = model.bind_tools([search_tables]).invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode([search_tables]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
