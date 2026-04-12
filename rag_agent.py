"""
RAG agent based on the LangChain/LangGraph tutorial:
docs/20260412-lecture-rags-by-langchain.md

Adapted to use Google Gemini (project default) instead of OpenAI.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# langchain-google-genai reads GOOGLE_API_KEY; alias GEMINI_API_KEY if needed
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ---------------------------------------------------------------------------
# 1. Preprocess documents
# ---------------------------------------------------------------------------
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# ---------------------------------------------------------------------------
# 2. Create a retriever tool
# ---------------------------------------------------------------------------
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.tools import tool

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
)
retriever = vectorstore.as_retriever()


@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_blog_posts

# ---------------------------------------------------------------------------
# 3–6. Nodes: generate_query_or_respond, grade_documents, rewrite_question,
#              generate_answer
# ---------------------------------------------------------------------------
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI  # embeddings use local fastembed
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

response_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
grader_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response or decide to retrieve."""
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    if response.binary_score == "yes":
        return "generate_answer"
    return "rewrite_question"


REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer using retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 7. Assemble the graph
# ---------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END},
)

workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
