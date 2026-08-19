from abc import ABC, abstractmethod
import random


# ==========================================
# 1. Runnable Base Class
# ==========================================

class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass


# ==========================================
# 2. Fake LLM
# ==========================================

class NakliLLM(Runnable):

    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):

        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {
            "response": random.choice(response_list)
        }

    def predict(self, prompt):

        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {
            "response": random.choice(response_list)
        }


# ==========================================
# 3. Fake PromptTemplate
# ==========================================

class NakliPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):

        return self.template.format(**input_dict)

    def format(self, input_dict):

        return self.template.format(**input_dict)


# ==========================================
# 4. Fake String Output Parser
# ==========================================

class NakliStrOutputParser(Runnable):

    def __init__(self):
        pass

    def invoke(self, input_data):

        return input_data["response"]


# ==========================================
# 5. Runnable Connector
# ==========================================

class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:

            input_data = runnable.invoke(input_data)

        return input_data


# ==========================================
# 6. Simple Chain
# ==========================================

template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

llm = NakliLLM()

parser = NakliStrOutputParser()


# Chain:
# PromptTemplate -> LLM -> OutputParser

chain = RunnableConnector([
    template,
    llm,
    parser
])


result = chain.invoke({
    "length": "long",
    "topic": "india"
})

print("\nSimple Chain Output:")
print(result)


# ==========================================
# 7. First Chain
# ==========================================

template1 = NakliPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)


# ==========================================
# 8. Second Chain
# ==========================================

template2 = NakliPromptTemplate(
    template="Explain the following joke {response}",
    input_variables=["response"]
)


llm = NakliLLM()

parser = NakliStrOutputParser()


# Chain 1:
# Template1 -> LLM

chain1 = RunnableConnector([
    template1,
    llm
])


# Chain 2:
# Template2 -> LLM -> Parser

chain2 = RunnableConnector([
    template2,
    llm,
    parser
])


# ==========================================
# 9. Combine Chain 1 + Chain 2
# ==========================================

final_chain = RunnableConnector([
    chain1,
    chain2
])


# ==========================================
# 10. Invoke Final Chain
# ==========================================

result = final_chain.invoke({
    "topic": "cricket"
})


print("\nFinal Chain Output:")
print(result)