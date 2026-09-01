from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """ LangChain is a framework for developing applications powered by 
          large language models. It provides tools and components that make it easier 
          to build AI applications. Developers can use LangChain to work with language models,
          prompts, chains, agents, memory, document loaders, and retrieval systems.
       """

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print(result[0])
print(len(result))
