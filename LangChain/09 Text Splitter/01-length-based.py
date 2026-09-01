from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader


# text = """LangChain is a framework for developing applications powered by large language models. It provides tools and components that make it easier to build AI applications. Developers can use LangChain to work with language models, prompts, chains, agents, memory, document loaders, and retrieval systems.

# Text splitting is an important part of many LangChain applications. Large documents are usually divided into smaller chunks before they are converted into embeddings. These chunks can then be stored in a vector database and retrieved when a user asks a question.

# A good text splitter should create chunks that are small enough to process efficiently while keeping enough context to preserve the meaning of the original document. Chunk size and chunk overlap are two important parameters that control how the text is divided.

# LangChain provides several text splitters, including CharacterTextSplitter and RecursiveCharacterTextSplitter. RecursiveCharacterTextSplitter is commonly used because it tries to split text using paragraphs, lines, words, and finally individual characters while maintaining meaningful context."""


loader = PyPDFLoader('suly.pdf')

pdf = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

# result = splitter.split_text(text)
result = splitter.split_documents(pdf)

# print(result)
# print(result[0])
print(result[0].page_content)