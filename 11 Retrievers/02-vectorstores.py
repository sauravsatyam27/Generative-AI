import os
from dotenv import load_dotenv

from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Gemini
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

# Create context
context = "\n\n".join(doc.page_content for doc in docs)

# Ask Gemini
prompt = f"""
Answer the following question using the provided Wikipedia context.

Question:
{query}

Context:
{context}
"""

response = llm.invoke(prompt)

print("\n--- Gemini Answer ---")
print(response.content)