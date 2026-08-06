from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(## ChatGoogleGenerativeAI
    model="gemini-2.5-flash"
)

result = model.invoke("What is the capital of India?")

print(result)
print(result.content)