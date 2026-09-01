from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_core.documents import Document

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()


# --------------------------------------------------
# 1. Create Documents
# --------------------------------------------------

all_docs = [

    Document(
        page_content="""
        Regular exercise can improve energy levels.
        Walking, running, cycling and strength training
        can help improve physical fitness and energy.
        """
    ),

    Document(
        page_content="""
        Getting enough sleep is important for maintaining
        energy throughout the day. Adults generally need
        consistent and quality sleep.
        """
    ),

    Document(
        page_content="""
        A balanced diet containing protein, carbohydrates,
        healthy fats, fruits and vegetables can support
        energy levels and overall health.
        """
    ),

    Document(
        page_content="""
        Drinking enough water is important because dehydration
        can cause tiredness and reduce concentration.
        """
    ),

    Document(
        page_content="""
        Meditation and mindfulness can help reduce stress
        and maintain mental balance.
        """
    ),

    Document(
        page_content="""
        Maintaining a healthy work-life balance can reduce
        stress and improve overall well-being.
        """
    ),

    Document(
        page_content="""
        Taking regular breaks during work can reduce mental
        fatigue and help maintain productivity.
        """
    )
]


# --------------------------------------------------
# 2. Gemini Embedding Model
# --------------------------------------------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# --------------------------------------------------
# 3. Create FAISS Vector Store
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)


# --------------------------------------------------
# 4. Create Similarity Retriever
# --------------------------------------------------

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# --------------------------------------------------
# 5. Gemini Chat Model
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# --------------------------------------------------
# 6. Create MultiQuery Retriever
# --------------------------------------------------

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={
            "k": 5
        }
    ),
    llm=llm
)


# --------------------------------------------------
# 7. User Query
# --------------------------------------------------

query = "How to improve energy levels and maintain balance?"


# --------------------------------------------------
# 8. Similarity Search
# --------------------------------------------------

similarity_results = similarity_retriever.invoke(query)


# --------------------------------------------------
# 9. MultiQuery Search
# --------------------------------------------------

multiquery_results = multiquery_retriever.invoke(query)


# --------------------------------------------------
# 10. Print Similarity Results
# --------------------------------------------------

print("\n")
print("=" * 80)
print("SIMILARITY RETRIEVER RESULTS")
print("=" * 80)

for i, doc in enumerate(similarity_results):

    print(f"\n--- Result {i + 1} ---")

    print(doc.page_content)


# --------------------------------------------------
# 11. Print MultiQuery Results
# --------------------------------------------------

print("\n")
print("=" * 80)
print("MULTIQUERY RETRIEVER RESULTS")
print("=" * 80)

for i, doc in enumerate(multiquery_results):

    print(f"\n--- Result {i + 1} ---")

    print(doc.page_content)