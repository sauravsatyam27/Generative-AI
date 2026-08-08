from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)

load_dotenv()

# Define the model

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# Define response schema

schema = [
    ResponseSchema(
        name="fact_1",
        description="Fact 1 about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="Fact 2 about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="Fact 3 about the topic"
    )
]


# Create structured output parser

parser = StructuredOutputParser.from_response_schemas(schema)


# Create prompt

template = PromptTemplate(
    template="""
Give 3 facts about {topic}.

{format_instruction}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


# Generate prompt

# prompt = template.invoke({
#     "topic": "Virat Kohli"
# })


# # Call model

# result = model.invoke(prompt)


# # Parse output

# final_result = parser.parse(result.content)


# print(final_result)

chain = template | model | parser

result = chain.invoke({"topic": "Virat Kohli"})

print(result)

## It can give data validation