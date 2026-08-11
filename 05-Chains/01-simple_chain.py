from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"

)

prompt = PromptTemplate(
    template='Write any 5 important Facts about {topic}',
    input_variables=['topic']
) 

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic' : 'Cricket'})

print(result)

chain.get_graph().print_ascii()