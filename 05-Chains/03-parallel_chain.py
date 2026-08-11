from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate detailed notes on the topic {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 10 quiz questions from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""Merge the following notes and quiz into a single document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser,
})

chain = parallel_chain | prompt3 | model1 | parser


# Input text
text = """
Cricket is one of the most popular sports in the world.
It is played between two teams, with each team consisting
of eleven players. The game involves batting, bowling,
fielding and scoring runs.
"""


result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()