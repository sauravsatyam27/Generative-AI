from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load .env
load_dotenv()


# -----------------------------
# 1. Create Documents
# -----------------------------

documents = [
    Document(
        page_content="Virat Kohli is one of the highest run scorers in IPL."
    ),

    Document(
        page_content="Virat Kohli has scored many memorable runs in IPL."
    ),

    Document(
        page_content="Virat Kohli has played many memorable IPL innings."
    ),

    Document(
        page_content="Virat Kohli has served as captain of Royal Challengers Bengaluru."
    ),

    Document(
        page_content="Virat Kohli is known for his aggressive batting style."
    ),

    Document(
        page_content="Virat Kohli has won the Orange Cap in IPL."
    )
]


# -----------------------------
# 2. Create Gemini Embeddings
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# -----------------------------
# 3. Create Chroma Vector Store
# -----------------------------

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)


# -----------------------------
# 4. Create MMR Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",

    search_kwargs={
        "k": 3,                # <-- This enables MMR
        "fetch_k": 6,
        "lambda_mult": 0.5     # k = top results, lambda_mult = relevance-diversity balance
    }
)


# -----------------------------
# 5. User Query
# -----------------------------

query = "Virat Kohli IPL career"


# -----------------------------
# 6. Retrieve Documents
# -----------------------------

results = retriever.invoke(query)


# -----------------------------
# 7. Print Results
# -----------------------------

for i, doc in enumerate(results, 1):

    print(f"\nDocument {i}")
    print("----------------")
    print(doc.page_content)