from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# JSON Output Parser
parser = JsonOutputParser()

# Prompt Template
template = PromptTemplate(
    template="""
Give me the name, age and city of a fictional person.

{format_instruction}
""",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions() ## Return a JSON object.
    }
)

prompt = template.format()

print(prompt)

chain = template | model | parser

# result = model.invoke(prompt)

# print(result)

# final_result = parser.parse(result.content)

# print(final_result)

# print(type(final_result))


result = chain.invoke({})

print(result)

# we caan not get a strcuted output in JSON output Parser  