#Import
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama


#definisco un input generico di test
test_input = "Hello, who are you?"

#definisco l'url del server Ollama
ollama_url = "http://localhost:11434"

#definisco il modello di chat
model_llm = "assistente"


#definisco i parametri del modello di chat su ollama
llm = ChatOllama(
    model = model_llm,
    temperature = 0.5,
    base_url = ollama_url,
    verbose = False
    )


#definisco una f che prende un input str e lo passa al modello di chat su ollama, restituendo un output str
def chat_with_llm (input: str = test_input) -> str:
    risposta = llm.invoke(test_input)
    return risposta


if __name__ == "__main__":
    #testo la f
    output = chat_with_llm()
    print(output)