from langchain_community.tools import DuckDuckGoSearchRun

searchTool = DuckDuckGoSearchRun()

while True:

    question = input("Enter Your Question: ")

    result = searchTool.invoke(question)

    print(result)