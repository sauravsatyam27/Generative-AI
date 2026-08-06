# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)

# result = model.invoke("Who is Captain of indian Cricket Team");

# print(result.content)

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    provider="auto",
    task="text-generation",
    max_new_tokens=256,
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("Who is the captain of the Indian cricket team?")

print(response.content)