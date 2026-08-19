from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
)


prompt1 = PromptTemplate(
    template='Generate a twitter post about the {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Generate a Linkdin post about the {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkdin' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic' : 'Love'})

print(result)