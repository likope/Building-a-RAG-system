#Import
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama


#definisco un input generico di test
test_input = "Hello, who are you?"

#definisco l'url del server Ollama
ollama_url = "http://localhost:11434"

#definisco il modello di chat
model_llm = "assistente"

#definisco la dict state che conterrà i parametri della conversazione
state = {
    "input" : "user input",
    "output": "model output",
    "history": "conversation history"
}


#definisco i parametri del modello di chat su ollama
llm = ChatOllama(
    model = model_llm,
    temperature = 0.5,
    base_url = ollama_url,
    )


#definisco una f che prende un input str e lo passa al modello di chat su ollama, restituendo un output str
def chat_with_llm (input: str = test_input) -> str:
    risposta = llm.invoke(test_input)
    return risposta


#definisco una f che prende l input e output e costruisce la history relativa nel state
def update_history(state: dict = state) -> dict:
    return state["history"] + "User input: " + state["input"] + "Model output: " + state["output"]


if __name__ == "__main__":
    #testo la f
    chain = RunnableLambda(chat_with_llm) | RunnableLambda(update_history)
    print(state)