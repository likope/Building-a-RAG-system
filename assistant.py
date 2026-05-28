from client import params_llm
from time import sleep
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser # Consigliato per estrarre il testo dall'LLM

class Assistente:
    
    def __init__(self):
        # Manteniamo solo la storia a lungo termine come stringa
        self.history = ""
        self.llm = params_llm()
        self.prompt_template = """Il tuo compito è quello di rispondere nel modo piu coerente possibile all input in base alla storia della conversazione e alle indicazioni.
        input = {input}
        storia = {history}"""
        
        self.prompt = PromptTemplate.from_template(self.prompt_template)

    def format_history(self) -> str:
        return self.history

    def append_to_history(self, current_input: str, model_output: str):
        """Aggiorna la storia della classe dopo che l'LLM ha risposto."""
        self.history += f"input = {current_input}, output_llm = {model_output}; "

    def Ask(self, user_input: str):
        print("Prompt ricevuto!")
        
        # Prepariamo lo stato iniziale per questa specifica chiamata
        current_state = {"input": user_input}
        
        print("Costruzione e invocazione della catena...")
        # 1. Usiamo .assign() PRIMA del prompt per iniettare la storia nel dizionario
        # 2. StrOutputParser trasforma la risposta dell'LLM in una stringa pulita
        chain = (
            RunnablePassthrough.assign(history=self.format_history) 

            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Eseguiamo la catena passandogli lo stato corrente
        output = chain.invoke(current_state)
        
        # Ora che abbiamo la risposta, aggiorna la storia persistente della classe
        self.append_to_history(user_input, output)
        
        return output