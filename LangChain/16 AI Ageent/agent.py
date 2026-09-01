import os
import requests

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


# =====================================================
# 1. Load environment variables
# =====================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# =====================================================
# 2. Search Tool
# =====================================================

search_tool = DuckDuckGoSearchRun()


# =====================================================
# 3. Weather Tool
# =====================================================

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data
    for a given city.
    """

    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key=4d1d8ae207a8c845a52df8a67bf3623e"
        f"&query={city}"
    )

    response = requests.get(url)

    return response.text


# =====================================================
# 4. Gemini LLM
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# =====================================================
# 5. Get ReAct Prompt
# =====================================================

prompt = hub.pull("hwchase17/react")


# =====================================================
# 6. Create ReAct Agent
# =====================================================

tools = [
    search_tool,
    get_weather_data
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# =====================================================
# 7. Agent Executor
# =====================================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# =====================================================
# 8. Invoke Agent
# =====================================================

response = agent_executor.invoke({
    "input": "Find the capital of Madhya Pradesh, then find its current weather condition"
})


# =====================================================
# 9. Final Output
# =====================================================

print(response["output"])