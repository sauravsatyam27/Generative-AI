from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


llm_with_tools = llm.bind_tools([add])


llm.bind_tools([add])


llm.bind_tools([add])

response = llm_with_tools.invoke(
    "What is 10 + 20?"
)

print(response)
