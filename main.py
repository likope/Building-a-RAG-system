#Import
from langchain_core.runnables import RunnableLambda, RunnableAssign
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from config import state, prompt, llm
from f import update_history



#determino la chain
chain = (
    RunnableAssign({"output": prompt | llm | StrOutputParser() }) | RunnableAssign({"history": update_history})
)


if __name__ == "__main__":
    while True:
        user_input = input("Scrivi il prompt o digita 'exit' per uscire: \n")
        if user_input.lower() == "exit":
            break
        state["input"] = user_input
        result = chain.invoke(state)
        print(result)