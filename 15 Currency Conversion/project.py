import os
import json
import requests

from dotenv import load_dotenv
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# ==========================================
# Tool 1: Get Currency Conversion Factor
# ==========================================

@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> str:
    """
    Fetch the currency conversion factor between
    a base currency and target currency.
    """

    api_key = os.getenv("EXCHANGE_RATE_API_KEY")

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{api_key}/pair/{base_currency}/{target_currency}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return "Unable to fetch exchange rate."

    data = response.json()

    return json.dumps(data)


# ==========================================
# Tool 2: Convert Currency
# ==========================================

@tool
def convert(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    Given a currency conversion rate, calculate
    the target currency value from a base currency value.
    """

    return base_currency_value * conversion_rate


# ==========================================
# Test Tools
# ==========================================

result = get_conversion_factor.invoke({
    "base_currency": "USD",
    "target_currency": "INR"
})

print("Conversion factor:")
print(result)


result = convert.invoke({
    "base_currency_value": 10,
    "conversion_rate": 85.16
})

print("Converted value:")
print(result)


# ==========================================
# Gemini Model
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# ==========================================
# Bind Tools
# ==========================================

llm_with_tools = llm.bind_tools([
    get_conversion_factor,
    convert
])


# ==========================================
# User Query
# ==========================================

messages = [
    HumanMessage(
        content=(
            "What is the conversion factor between INR and USD, "
            "and based on that can you convert 10 INR to USD?"
        )
    )
]


# ==========================================
# First LLM Call
# ==========================================

ai_message = llm_with_tools.invoke(messages)

print("\nAI Tool Calls:")
print(ai_message.tool_calls)

messages.append(ai_message)


# ==========================================
# Execute Tools
# ==========================================

conversion_rate = None

for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "get_conversion_factor":

        tool_result = get_conversion_factor.invoke(
            tool_call["args"]
        )

        print("\nConversion API Result:")
        print(tool_result)

        data = json.loads(tool_result)

        conversion_rate = data["conversion_rate"]

        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )
        )

    elif tool_call["name"] == "convert":

        if conversion_rate is None:
            continue

        tool_result = convert.invoke({
            "base_currency_value":
                tool_call["args"]["base_currency_value"],
            "conversion_rate":
                conversion_rate
        })

        print("\nConverted Value:")
        print(tool_result)

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )


# ==========================================
# Final Response
# ==========================================

final_response = llm_with_tools.invoke(messages)

print("\nFinal Answer:")
print(final_response.content)