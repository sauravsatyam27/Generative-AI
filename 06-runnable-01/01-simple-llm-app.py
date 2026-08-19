from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Define the input
topic = input("Enter a topic: ")

# Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Call Gemini directly
response = llm.invoke(formatted_prompt)

# Print the output
print("Generated Blog Title:", response.content)