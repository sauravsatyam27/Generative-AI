from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return their product."""
    return a * b


llm_with_tool = model.bind_tools([multiply])


# User message
query = HumanMessage(
    content="Can you multiply 2 and 5?"
)

messages = [query]


# LLM decides whether to call the tool
result = llm_with_tool.invoke(messages)

print("Tool call:")
print(result.tool_calls)


# Add AI response to conversation
messages.append(result)


# Get arguments
args = result.tool_calls[0]["args"]

print("Arguments:")
print(args)


# Execute tool
tool_result = multiply.invoke(args)

print("Tool result:")
print(tool_result)


# Convert tool output into ToolMessage
tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=result.tool_calls[0]["id"]
)

# Add tool result to conversation
messages.append(tool_message)


# Send complete conversation back to LLM
final_result = llm_with_tool.invoke(messages)

print("Final answer:")
print(final_result.content)