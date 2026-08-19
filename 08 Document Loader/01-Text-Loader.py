from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write the summary of the {poem}',
    input_variables=['poem']
)

loader = TextLoader('poem.txt')

docs = loader.load()

# print(docs)
# print(type(docs))


chain = prompt | model | parser

result = chain.invoke({
    'poem' : docs[0].page_content
})


print(result)
# print(len(docs))

# print(docs[0])
# print(docs[0].page_content)


## it return a list
