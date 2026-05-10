from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


user_input = ""

#definisco la dict state che conterrà i parametri della conversazione
state = {
    "input" : user_input,
    "output": "",
    "history": ""
}

prompt = PromptTemplate.from_template(f"user input: {state['input']}, history: {state['history']}")

#definisco i parametri del modello di chat su ollama
llm = ChatOllama(
        model = "assistente",
        temperature = 0.2,
        base_url = "http://localhost:11434"
    )