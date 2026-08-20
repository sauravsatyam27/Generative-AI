import os
from dotenv import load_dotenv

from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Wikipedia Retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "The geopolitical history of India and Pakistan from the perspective of a Chinese"

# Retrieve documents
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:1000])