import os
import json
import requests

from dotenv import load_dotenv
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. Load Environment Variables
# ============================================================

load_dotenv()

EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not EXCHANGE_RATE_API_KEY:
    raise ValueError("EXCHANGE_RATE_API_KEY not found in .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# ============================================================
# Tool 1: Get Currency Conversion Factor
# ============================================================

@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> str:
    """
    Fetch the currency conversion factor between
    a base currency and target currency.
    """

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{EXCHANGE_RATE_API_KEY}/pair/"
        f"{base_currency}/{target_currency}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return "Unable to fetch exchange rate."

    data = response.json()

    if data.get("result") != "success":
        return "Unable to fetch exchange rate."

    return json.dumps(data)


# ============================================================
# Tool 2: Convert Currency
# ============================================================

@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    Given a currency conversion rate, calculate
    the target currency value from a base currency value.
    """

    return base_currency_value * conversion_rate


# ============================================================
# Test Tool 1
# ============================================================

print("=" * 60)
print("TESTING CURRENCY TOOLS")
print("=" * 60)

result = get_conversion_factor.invoke({
    "base_currency": "USD",
    "target_currency": "INR"
})

print("\nConversion factor API response:")
print(result)


# ============================================================
# Test Tool 2
# ============================================================

result = convert.invoke({
    "base_currency_value": 10,
    "conversion_rate": 85.16
})

print("\nConverted value:")
print(result)


# ============================================================
# Gemini Model
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# Bind Tools
# ============================================================

llm_with_tools = llm.bind_tools([
    get_conversion_factor,
    convert
])


# ============================================================
# User Query
# ============================================================

messages = [
    HumanMessage(
        content=(
            "What is the conversion factor between INR and USD, "
            "and based on that can you convert 10 INR to USD?"
        )
    )
]


# ============================================================
# First LLM Call
# ============================================================

print("\n" + "=" * 60)
print("FIRST LLM CALL")
print("=" * 60)

ai_message = llm_with_tools.invoke(messages)

print("\nAI Tool Calls:")

for tool_call in ai_message.tool_calls:
    print(tool_call)

messages.append(ai_message)


# ============================================================
# Execute Tool Calls
# ============================================================

conversion_rate = None

for tool_call in ai_message.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_call_id = tool_call["id"]

    # --------------------------------------------------------
    # Tool 1: Get Conversion Factor
    # --------------------------------------------------------

    if tool_name == "get_conversion_factor":

        print("\n" + "-" * 60)
        print("Calling get_conversion_factor")
        print("-" * 60)

        tool_result = get_conversion_factor.invoke(tool_args)

        print("\nConversion API Result:")
        print(tool_result)

        # Convert JSON string into Python dictionary
        data = json.loads(tool_result)

        # Extract conversion rate
        conversion_rate = data.get("conversion_rate")

        print("\nConversion Rate:")
        print(conversion_rate)

        # Send result back to LLM
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call_id
            )
        )

    # --------------------------------------------------------
    # Tool 2: Convert Currency
    # --------------------------------------------------------

    elif tool_name == "convert":

        print("\n" + "-" * 60)
        print("Calling convert")
        print("-" * 60)

        if conversion_rate is None:
            print("Conversion rate not available.")
            continue

        base_currency_value = tool_args["base_currency_value"]

        tool_result = convert.invoke({
            "base_currency_value": base_currency_value,
            "conversion_rate": conversion_rate
        })

        print("\nConverted Value:")
        print(tool_result)

        # Send result back to LLM
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )
        )


# ============================================================
# Final LLM Call
# ============================================================

print("\n" + "=" * 60)
print("FINAL LLM CALL")
print("=" * 60)

final_response = llm_with_tools.invoke(messages)


# ============================================================
# Final Answer
# ============================================================

print("\nFinal Answer:")
print(final_response.content)