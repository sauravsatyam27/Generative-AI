from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)




## Phase 1 : PDF loading
loader = PyPDFLoader("The IPL.pdf")
pdf = loader.load()
# print(len(pdf[0].page_content))


## Phase 2 : Text Splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)


chunks = splitter.split_documents(pdf)

# print("Pages:", len(pdf))
# print("Chunks:", len(chunks))


## Phase 3 : Converting pdf into embeddings

## Embedding model , -> data ko embeddings me convert karne ke liye
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)




vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample"   
)



vector_store.add_documents(chunks)


chat_history = [
    SystemMessage(content="You are a Helpfull AI Assistent Chat Bot")
]



prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant.

Answer the user's question ONLY using the context provided below.

If the answer is not present in the context, say:
"Sorry, I couldn't find this information in the PDF."

Context:
{context}

Question:
{question}
""")


def correct_query(query):

    response = model.invoke(
        f"""
You are correcting a user's question for an IPL PDF search.

Fix:
- spelling mistakes
- typing mistakes
- missing letters
- obvious misspelled words
- grammar mistakes

Use cricket/IPL context to understand the intended word.

Examples:
rus -> runs
scord -> scored
kohlii -> kohli
virat kohlii -> virat kohli
how meny -> how many

Do NOT change the meaning.

Return ONLY the corrected question.
Do not explain anything.

User question:
{query}
"""
    )

    return response.content.strip()




while True:
    user_input = input("You 😊 : ")

    # Exit
    if user_input.lower() == "exit":
        break

    # 1. Correct spelling / grammar
    corrected_query = correct_query(user_input)

    print("Corrected Query:", corrected_query)

    # 2. Search using corrected query
    result = vector_store.similarity_search_with_score(
        corrected_query,
        k=6
    )

    # 3. Get best document and score
    best_doc, best_score = result[0]

  #  print("Score:", best_score)

    # 4. Check relevance
    if best_score < 0.7:

        # 5. Create context
        context = "\n\n".join(
            doc.page_content
            for doc, score in result
        )

        print("\n===== CONTEXT =====")
     #   print(context)
        print("==================")

        # 6. Create prompt
        messages = prompt.format_messages(
            context=context,
            question=corrected_query
        )

        # 7. Ask Gemini
        response = model.invoke(messages)

        # 8. Print answer
        print("\nAI 🤖 :", response.content)

    else:
        print(
            "\nAI 🤖 : Sorry, "
            "I couldn't find this information in the PDF."
        )