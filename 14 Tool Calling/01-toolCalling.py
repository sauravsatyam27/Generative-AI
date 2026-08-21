from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

@tool
def multiply(a : int, b: int)-> int:
    """ Given Two number Return their product """
    return a * b


print(multiply.invoke({'a': 3, 'b': 4}))

print(multiply.args)

llm_with_tool = model.bind_tools([multiply])