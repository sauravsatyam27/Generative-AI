from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

# Gemini
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 0. ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. Add it to your .env file."
    )


# ============================================================
# 1. GEMINI LLM + EMBEDDINGS
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# 2. PDF RETRIEVER STORE
# ============================================================

# Each chat thread has its own PDF retriever.
_THREAD_RETRIEVERS: Dict[str, Any] = {}

# Metadata for uploaded PDFs.
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """
    Fetch the retriever for a thread if available.
    """

    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    return None


# ============================================================
# 3. PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF
    and store it for the current chat thread.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    # --------------------------------------------------------
    # Create temporary PDF file
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        # ----------------------------------------------------
        # Split documents
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(docs)

        # ----------------------------------------------------
        # Create FAISS vector store using Gemini embeddings
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4
            },
        )

        # ----------------------------------------------------
        # Store retriever for this thread
        # ----------------------------------------------------

        thread_id = str(thread_id)

        _THREAD_RETRIEVERS[thread_id] = retriever

        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        # ----------------------------------------------------
        # Return information to Streamlit
        # ----------------------------------------------------

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:

        # Temporary PDF is no longer needed.
        try:
            os.remove(temp_path)

        except OSError:
            pass


# ============================================================
# 4. TOOLS
# ============================================================


# ------------------------------------------------------------
# Web Search
# ------------------------------------------------------------

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ------------------------------------------------------------
# Calculator
# ------------------------------------------------------------

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform a basic arithmetic operation.

    Supported operations:
    - add
    - sub
    - mul
    - div
    """

    try:

        if operation == "add":

            result = first_num + second_num

        elif operation == "sub":

            result = first_num - second_num

        elif operation == "mul":

            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed"
                }

            result = first_num / second_num

        else:

            return {
                "error": f"Unsupported operation '{operation}'"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ------------------------------------------------------------
# Stock Price
# ------------------------------------------------------------

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a symbol
    using Alpha Vantage.

    Example:
        AAPL
        TSLA
        MSFT
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is not configured."
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    try:

        response = requests.get(
            url,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        return {
            "error": str(e)
        }


# ------------------------------------------------------------
# PDF RAG Tool
# ------------------------------------------------------------

@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Retrieve relevant information from the PDF uploaded
    in the current chat thread.

    Always include the thread_id when calling this tool.
    """

    retriever = _get_retriever(thread_id)

    # --------------------------------------------------------
    # No PDF
    # --------------------------------------------------------

    if retriever is None:

        return {
            "error": (
                "No document indexed for this chat. "
                "Upload a PDF first."
            ),
            "query": query,
        }

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    result = retriever.invoke(query)

    context = [
        doc.page_content
        for doc in result
    ]

    metadata = [
        doc.metadata
        for doc in result
    ]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA
        .get(str(thread_id), {})
        .get("filename"),
    }


# ============================================================
# 5. TOOL LIST
# ============================================================

tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]


# ============================================================
# 6. BIND TOOLS TO GEMINI
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 7. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


# ============================================================
# 8. CHAT NODE
# ============================================================

def chat_node(
    state: ChatState,
    config=None,
):
    """
    Gemini decides whether to:
    - answer directly
    - call a tool
    """

    thread_id = None

    # --------------------------------------------------------
    # Get thread ID from LangGraph config
    # --------------------------------------------------------

    if config and isinstance(config, dict):

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id")
        )

    # --------------------------------------------------------
    # System instructions
    # --------------------------------------------------------

    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant.\n\n"

            "You have access to several tools.\n\n"

            "1. PDF RAG:\n"
            "If the user asks about information from "
            "their uploaded PDF, use the `rag_tool`.\n"
            f"The current thread_id is `{thread_id}`.\n"
            "Always provide this thread_id when using rag_tool.\n\n"

            "2. Web Search:\n"
            "Use the web search tool when current or "
            "external information is required.\n\n"

            "3. Stock Price:\n"
            "Use the stock price tool when the user asks "
            "for stock information.\n\n"

            "4. Calculator:\n"
            "Use the calculator tool for arithmetic.\n\n"

            "If the user asks about a PDF but no PDF has "
            "been uploaded, tell the user to upload a PDF first."
        )
    )

    # --------------------------------------------------------
    # Add system message
    # --------------------------------------------------------

    messages = [
        system_message,
        *state["messages"],
    ]

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# 9. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 10. SQLITE CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(
    conn=conn
)


# ============================================================
# 11. LANGGRAPH
# ============================================================

graph = StateGraph(ChatState)


# Nodes
graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_node(
    "tools",
    tool_node,
)


# ------------------------------------------------------------
# START -> CHAT
# ------------------------------------------------------------

graph.add_edge(
    START,
    "chat_node",
)


# ------------------------------------------------------------
# CHAT -> TOOL or END
# ------------------------------------------------------------

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)


# ------------------------------------------------------------
# TOOL -> CHAT
# ------------------------------------------------------------

graph.add_edge(
    "tools",
    "chat_node",
)


# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 12. THREAD HELPERS
# ============================================================

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = (
            checkpoint
            .config["configurable"]
            ["thread_id"]
        )

        all_threads.add(thread_id)

    return list(all_threads)


# ============================================================
# 13. CHECK WHETHER THREAD HAS PDF
# ============================================================

def thread_has_document(
    thread_id: str,
) -> bool:

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


# ============================================================
# 14. GET PDF METADATA
# ============================================================

def thread_document_metadata(
    thread_id: str,
) -> dict:

    return _THREAD_METADATA.get(
        str(thread_id),
        {},
    )