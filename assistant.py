from client_assistente import params_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class Assistant:

    def __init__(self):
        """
        Costruttore statico dell assistente, inizializza tutte le variabili e costruisce il template del prompt.
        """
        self.history = ""
        self.output = ""
        self.judge_output = ""
        self.llm = params_llm()
        self.prompt_template = """Il tuo compito è quello di rispondere nel modo piu coerente possibile all input in base alla storia della conversazione e alle indicazioni.
        input = {input}
        storia = {history}"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
    
    def get_actual_history(self, _):
        """
        f che restituisce l'attuale storia della conversazione.
        """
        return self.history
    
    def append_history(self, user_input: str):
        """
        f che aggiorna l attuale storia della conversazione.
        """
        self.history += f"input = {user_input}, " + f"output = {self.output}, " + f"judge_output = {self.judge_output}"
        return self.history
    
    def reset_history(self):
        """
        f che resetta la storia della conversazione.
        """
        self.history = ""
        return self.history
    
    def update_current_state(self, user_input: str):
        """
        f che si occupa di aggiornare lo stato della dict da passare al llm.
        """

        self.current_state = {
            "input": user_input,
            "history": self.history,
            "output": self.output,
            "judge_output": self.judge_output
        }

    def Ask(self, user_input: str, history_summary: str, risposta_giudice: str):
        """
        f che gestisce la creazione del prompt e la risposta del llm.
        """

        self.judge_output = risposta_giudice
        self.update_current_state(user_input)
        print(f"selfcurrentstate = {self.current_state}") #debug, parametri che vengono passati al modello llm

        if history_summary != "":
            self.history = history_summary

        self.chain = (
            RunnablePassthrough.assign(history = self.get_actual_history)
            |self.prompt
            |self.llm
            |StrOutputParser()
        )

        self.output = self.chain.invoke(self.current_state)
        
        self.history = self.append_history(user_input)
        self.update_current_state(user_input)
        return self.output, self.current_state
    
    def get_history_summary(self):
        """
        f che gestisce il riassunto della storia della conversazione tramite l llm.
        """
        self.template_history_summary = """Il tuo compito è quello di creare un riassunto della storia di una conversazione per un altra llm.
        storia = {history}."""
        self.prompt_history_summary = PromptTemplate.from_template(self.template_history_summary)
        self.history_summary_chain = self.prompt_history_summary | self.llm | StrOutputParser()
        self.history_summary = self.history_summary_chain.invoke(self.current_state)
        return self.history_summary