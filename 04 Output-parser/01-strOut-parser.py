from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# First prompt
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)


# Second prompt
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"]
)


# First LLM call
prompt1 = template1.invoke({
    "topic": "Black Hole"
})

result = model.invoke(prompt1)

print("FIRST OUTPUT:")
print(result.content)


# Second LLM call
prompt2 = template2.invoke({
    "text": result.content
})

result = model.invoke(prompt2)

print("\nFINAL OUTPUT:")
print(result.content)