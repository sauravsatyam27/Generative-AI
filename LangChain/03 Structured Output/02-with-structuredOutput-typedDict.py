from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

# Schema

class Review(TypedDict):
    summary : Annotated[str,"A breif summary of the review"]   # name should be a string.
    ## sentiments : str
    sentiments : Annotated[str,"Return Sentiments of the review"]

structured_output = model.with_structured_output(Review)

result = structured_output.invoke("""The hardware is great, but the software feels bloated. There are
too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to
other brands. Hoping for a software update to fix this. """)

print(result)

print(type(result))