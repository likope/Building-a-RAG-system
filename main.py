from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableAssign


user_input=""

state = {
    "input": user_input,
    "output": "",
    "history": "",
    "first_response": "",
    "second_response": ""
}

template = """Sei un assistente, rispondi nel modo più coerente possibile all'input dell'utente (user_input) tenendo conto anche della cronologia della conversazione (history).
user_input: {input},
history: {history}"""
prompt = PromptTemplate.from_template(template, template_format="f-string")

llm = ChatOllama(
    model = "assistente",
    temperature = 0.6,
    base_url = "http://localhost:11434"
)

def history_update(state: dict) -> str:
    prev = state["history"]
    new_turn = f"user: {state['input']}\nassistant: {state['output']}"
    if prev == "":
        return new_turn
    return prev + "\n" + new_turn

chain = RunnableAssign({"output": prompt|llm|StrOutputParser()}) | RunnableAssign({"history": history_update})


if __name__ == "__main__":
    i=0
    while True:
        user_input = input("Inserisci il prompt, digita 'exit' per terminare\n")
        state["input"] = user_input
        state = chain.invoke(state)
        print(state)