from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Tell me joke about {topic}',
    input_variables=['topic']
)

def wordCount(prompt1):
    return len(prompt1.split())




joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
     'joke' : RunnablePassthrough(),
     'word_count' : RunnableLambda(wordCount)
    }
)


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))