from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
)

prompt = PromptTemplate(
    template= 'Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)

print(chain.invoke({'topic' : 'Sex'}))