from langchain_core.runnables import RunnableLambda

def wordCounter(text):
    return len(text.split())


runnable_word_counter = RunnableLambda(wordCounter)

x = runnable_word_counter.invoke('Hii how are you')
print(x)