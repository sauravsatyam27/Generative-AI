from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

import os

load_dotenv()

# Gemini API Key
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"


# -----------------------------
# Create Gemini Embeddings
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# -----------------------------
# Create LangChain Documents
# -----------------------------

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]


# -----------------------------
# Create Chroma Vector Store
# -----------------------------

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample"
)


# -----------------------------
# Add Documents
# -----------------------------

vector_store.add_documents(docs)


# -----------------------------
# View Documents
# -----------------------------

# print(
#     vector_store.get(
#         include=["embeddings", "documents", "metadatas"]
#     )
# )


# -----------------------------
# Similarity Search
# -----------------------------

result = vector_store.similarity_search(
    query="Who among these are a bowler?",
    k=2
)

for doc in result:
    print(doc)


# -----------------------------
# Similarity Search + Score
# -----------------------------

# result = vector_store.similarity_search_with_score(
#     query="Who among these are a bowler?",
#     k=2
# )

# for doc, score in result:
#     print("Document:", doc)
#     print("Score:", score)


# -----------------------------
# Metadata Filtering
# -----------------------------

# result = vector_store.similarity_search(
#     query="",
#     k=2,
#     filter={"team": "Chennai Super Kings"}
# )

# for doc in result:
#     print(doc)


# -----------------------------
# Update Document
# -----------------------------

# updated_doc1 = Document(
#     page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances.",
#     metadata={"team": "Royal Challengers Bangalore"}
# )

# IMPORTANT:
# Replace this ID with the actual ID returned
# when you added the document.

# vector_store.update_document(
#     document_id="YOUR_DOCUMENT_ID",
#     document=updated_doc1
# )


# -----------------------------
# Delete Document
# -----------------------------

# vector_store.delete(
#     ids=["YOUR_DOCUMENT_ID"]
# )