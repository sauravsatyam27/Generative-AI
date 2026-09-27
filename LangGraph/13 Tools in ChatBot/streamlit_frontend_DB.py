import streamlit as st
from langrapgh_tool_backend import chatbot, retrieve_all_threads

from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ******************************** Utility Functions *************************


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# ******************************** Content Helper ****************************


def extract_text(content):
    """
    Extract only text from Gemini/LangChain message content.
    """

    # Normal string response
    if isinstance(content, str):
        return content

    # Gemini may return a list of content blocks
    if isinstance(content, list):

        text = ""

        for block in content:

            # Example:
            # {"type": "text", "text": "Hello"}

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text += block.get("text", "")

                elif "text" in block:
                    text += str(block["text"])

            # Sometimes content can contain strings directly
            elif isinstance(block, str):
                text += block

        return text

    # Fallback
    return str(content)


# **************************************** Session Setup **********************


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


add_thread(st.session_state["thread_id"])


# **************************************** Sidebar UI *************************


st.sidebar.title("LangGraph Chatbot")


if st.sidebar.button("New Chat"):
    reset_chat()


st.sidebar.header("My Conversations")


for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(str(thread_id)):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"

            elif isinstance(msg, AIMessage):
                role = "assistant"

            else:
                # Ignore ToolMessage etc. in the UI
                continue

            content = extract_text(msg.content)

            temp_messages.append(
                {
                    "role": role,
                    "content": content
                }
            )

        st.session_state["message_history"] = temp_messages


# **************************************** Main UI ****************************


# Display conversation history

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# Chat input

user_input = st.chat_input("Type here")


if user_input:

    # --------------------------------------------------
    # 1. Display user message
    # --------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------
    # 2. LangGraph Configuration
    # --------------------------------------------------

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },

        "metadata": {
            "thread_id": st.session_state["thread_id"],
            "run_name": "chat_turn"
        }
    }


    # --------------------------------------------------
    # 3. Assistant Response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""


        # Stream LangGraph

        for message_chunk, metadata in chatbot.stream(

            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },

            config=CONFIG,

            stream_mode="messages"
        ):

            # Only process AI messages

            if isinstance(message_chunk, AIMessage):

                # Extract clean text
                text = extract_text(message_chunk.content)

                if text:

                    full_response += text

                    response_placeholder.markdown(
                        full_response
                    )


        # --------------------------------------------------
        # 4. Save Assistant Response
        # --------------------------------------------------

        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": full_response
            }
        )