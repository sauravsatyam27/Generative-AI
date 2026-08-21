import os
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Load Environment Variables
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file"
    )


# --------------------------------------------------
# 2. Imports
# --------------------------------------------------

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)


# --------------------------------------------------
# 3. Create Gemini Embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# --------------------------------------------------
# 4. Load Existing FAISS Vector Store
# --------------------------------------------------

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded successfully!")


# --------------------------------------------------
# 5. Create Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)

print("Retriever created successfully!")


# --------------------------------------------------
# 6. Format Retrieved Documents
# --------------------------------------------------

def format_docs(retrieved_docs):

    context_text = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return context_text


# --------------------------------------------------
# 7. Create Parallel Chain
# --------------------------------------------------

parallel_chain = RunnableParallel({

    "context": retriever | RunnableLambda(format_docs),

    "question": RunnablePassthrough()

})


# --------------------------------------------------
# 8. Test Parallel Chain
# --------------------------------------------------

question = "Who is Demis?"

result = parallel_chain.invoke(question)


# --------------------------------------------------
# 9. Print Result
# --------------------------------------------------

print("\n==============================")
print("PARALLEL CHAIN RESULT")
print("==============================")

print("\nQuestion:")
print(result["question"])

print("\nContext:")
print(result["context"])