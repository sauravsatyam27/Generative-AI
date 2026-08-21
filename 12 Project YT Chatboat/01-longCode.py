import os
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# --------------------------------------------------
# 2. Imports
# --------------------------------------------------

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate


# --------------------------------------------------
# 3. YouTube Video ID
# --------------------------------------------------

video_id = "7aEAS5E5vjg"


# --------------------------------------------------
# 4. Get YouTube Transcript
# --------------------------------------------------

try:

    ytt_api = YouTubeTranscriptApi()

    fetched_transcript = ytt_api.fetch(video_id)

    # Convert transcript snippets into plain text
    transcript = " ".join(
        snippet.text
        for snippet in fetched_transcript
    )

    print("Transcript fetched successfully!")
    print("Transcript length:", len(transcript))


except TranscriptsDisabled:

    print("No captions available for this video.")
    exit()


except Exception as e:

    print("Error while fetching transcript:")
    print(e)
    exit()


# --------------------------------------------------
# 5. Split Transcript into Chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = text_splitter.create_documents(
    [transcript]
)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 6. Create Gemini Embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# --------------------------------------------------
# 7. Create FAISS Vector Store
# --------------------------------------------------

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created successfully!")


# --------------------------------------------------
# 7.1 Save FAISS Vector Store
# --------------------------------------------------

vector_store.save_local("faiss_index")

print("FAISS vector store saved successfully!")


# --------------------------------------------------
# 8. Create Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


# --------------------------------------------------
# 9. Create Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)


# --------------------------------------------------
# 10. Prompt Template
# --------------------------------------------------

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY using the provided YouTube transcript context.

If the answer is not available in the context,
say:

"I don't know based on the provided transcript."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question"
    ]
)


# --------------------------------------------------
# 11. Ask Question
# --------------------------------------------------

question = """
Is the topic of nuclear fusion discussed in this video?
If yes, what was discussed?
"""


# --------------------------------------------------
# 12. Retrieve Relevant Documents
# --------------------------------------------------

retrieved_docs = retriever.invoke(question)

print("\nRetrieved Documents:")
print("--------------------")

for i, doc in enumerate(retrieved_docs, start=1):

    print(f"\nDocument {i}:")
    print(doc.page_content[:500])


# --------------------------------------------------
# 13. Create Context
# --------------------------------------------------

context_text = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)


# --------------------------------------------------
# 14. Create Final Prompt
# --------------------------------------------------

final_prompt = prompt.invoke(
    {
        "context": context_text,
        "question": question
    }
)


# --------------------------------------------------
# 15. Ask Gemini
# --------------------------------------------------

answer = llm.invoke(final_prompt)


# --------------------------------------------------
# 16. Print Final Answer
# --------------------------------------------------

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer.content)